"""Put an independently reviewed sofa into a copy of the recovered room."""
import argparse
import json
import logging
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from omegaconf import OmegaConf
from pydrake.all import RigidTransform, RollPitchYaw

from scenecode.agent_utils.house import HouseScene
from scenecode.agent_utils.room import ObjectType
from scenecode.agent_utils.mesh_physics_analyzer import MeshPhysicsAnalysis
from scenecode.agent_utils.sdf_generator import generate_drake_sdf, YUP_TO_ZUP_TRANSFORM
from scenecode.agent_utils.convex_decomposition_server.server_app import ConvexDecompositionServerApp
from scenecode.agent_utils.physical_feasibility import apply_physical_feasibility_postprocessing
from scripts.validate_qwen_room import validate


def integrate(source, repair, output):
    source, repair, output = source.resolve(), repair.resolve(), output.resolve()
    review = json.loads((repair/'independent_review.json').read_text())
    if review.get('passed') is not True:
        raise ValueError('Independent sofa review must pass before room integration')
    candidate = repair / review['selected_attempt']
    if (output/'room').exists():
        raise FileExistsError(output/'room')
    output.mkdir(parents=True,exist_ok=True)
    record={'kind':'reviewed_sofa_repair_integration_not_fresh_baseline', 'source':str(source),
            'repair':str(repair),'state':'running','started_at':time.time(),'new_model_calls':0}
    path=output/'integration.json'
    path.write_text(json.dumps(record,indent=2))
    try:
        shutil.copytree(source/'room',output/'room',ignore=shutil.ignore_patterns('house.blend','independent_validation.json'))
        shutil.copy2(source/'recovery_config.yaml',output/'recovery_config.yaml')
        shutil.copy2(repair/'independent_review.json',output/'sofa_independent_review.json')
        cfg=OmegaConf.to_container(OmegaConf.load(output/'recovery_config.yaml'),resolve=True)
        house_dir=output/'room/scene_000'
        state=json.loads((house_dir/'combined_house_after_furniture/house_state.json').read_text())
        house=HouseScene.from_state_dict(state,house_dir=house_dir)
        room=house.rooms['living_room']
        sofa=next(o for o in room.objects.values() if o.name=='sofa')
        asset_dir=room.scene_dir/'generated_assets/furniture/sdf/sofa_repaired'
        asset_dir.mkdir()
        scene=trimesh.load(candidate/'sofa.glb',force='scene')
        original_width=float(sofa.bbox_max[0]-sofa.bbox_min[0])
        scale=original_width/scene.extents[0]
        scene.apply_scale(scale)
        bounds=scene.bounds
        scene.apply_translation([-bounds[:,0].mean(),-bounds[0,1],-bounds[:,2].mean()])
        # Original Blender front -Y exports as GLTF +Z; canonical room front is +Y (GLTF -Z).
        scene.apply_transform(np.diag([-1.,1.,-1.,1.]))
        for filename, data in scene.export(file_type='gltf').items():
            (asset_dir/filename).write_bytes(data)
        visual=asset_dir/'model.gltf'
        mesh=scene.to_geometry()
        collision_cfg=cfg['furniture_agent']['collision_geometry']
        if collision_cfg['method']!='vhacd':
            raise ValueError('This recovery expects the original V-HACD configuration')
        settings=dict(collision_cfg['vhacd'])
        settings['vhacd_resolution']=settings.pop('resolution')
        pieces=ConvexDecompositionServerApp._run_vhacd(None,mesh,settings)
        collision=[trimesh.Trimesh(vertices=p['vertices'],faces=p['faces']) for p in pieces]
        mass=float(ET.parse(sofa.sdf_path).findtext('model/link/inertial/mass'))
        analysis=MeshPhysicsAnalysis(up_axis='+Z',front_axis='+Y',material='fabric',mass_kg=mass,mass_range_kg=(mass,mass))
        sdf=asset_dir/'sofa.sdf'
        generate_drake_sdf(visual,collision,analysis,sdf,asset_name='sofa_repaired')
        zup=mesh.copy()
        zup.apply_transform(YUP_TO_ZUP_TRANSFORM)
        sofa.geometry_path=visual
        sofa.sdf_path=sdf
        sofa.bbox_min,sofa.bbox_max=zup.bounds
        sofa.description='Light-gray three-seat sofa with three rectangular seat and back cushions and two padded side arms'
        sofa.metadata.update(repair_source=str(repair),repair_mesh=str(candidate/'sofa.glb'),
                             repair_spec='Original width, uniform scaling, known canonical axes; original mass and collision settings')
        yaw=RollPitchYaw(sofa.transform.rotation()).yaw_angle()
        position=sofa.transform.translation().copy()
        position[2]=0
        sofa.transform=RigidTransform(RollPitchYaw(0,0,yaw),position)
        projection=cfg['experiment']['projection']
        f,s=projection['furniture'],projection['simulation']
        phys=cfg['furniture_agent']['physics_validation']
        room, success, removed=apply_physical_feasibility_postprocessing(
            scene=room,weld_furniture=False,projection_enabled=projection['enabled'] and f['enabled'],
            projection_influence_distance=f['influence_distance'],projection_solver_name=f['solver_name'],
            projection_iteration_limit=f['iteration_limit'],projection_time_limit_s=f['time_limit_s'],
            projection_xy_only=f['xy_only'],projection_fix_rotation=f['fix_rotation'],
            simulation_enabled=s['enabled'],simulation_time_s=s['simulation_time_s'],simulation_time_step_s=s['time_step_s'],
            simulation_timeout_s=s['timeout_s'],simulation_html_path=None,
            remove_fallen_furniture=phys['remove_fallen_furniture'],fallen_tilt_threshold_degrees=phys['fallen_tilt_threshold_degrees'])
        house.rooms['living_room']=room
        house.assemble(cfg=None,output_name='combined_house_after_furniture',include_object_types=[ObjectType.FURNITURE])
        report=validate(output/'room')
        (output/'room/pre_export_validation.json').write_text(json.dumps(report,indent=2))
        errors=[e for e in report['errors'] if not e.startswith('Missing final export:') or not e.endswith('house.blend')]
        record.update(state='ready_for_gpu_export' if not errors else 'validation_failed',
                      scale_factor=scale,dimensions_m=(sofa.bbox_max-sofa.bbox_min).tolist(),
                      collision_pieces=len(collision),mass_kg=mass,removed_objects=[str(x) for x in removed],
                      pre_export_checks_passed=not errors,errors=errors)
        return not errors
    except Exception as exc:
        record.update(state='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_at']=time.time()
        path.write_text(json.dumps(record,indent=2))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--repair',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(0 if integrate(args.source,args.repair,args.output) else 1)
