"""Material-only assisted recovery; preserve triangle positions and all physics files."""
import argparse
import copy
import hashlib
import json
import shutil
import struct
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blacken_visual(path, whole=False):
    doc = json.loads(path.read_text())
    buffers = [(path.parent / b['uri']).read_bytes() for b in doc['buffers']]

    def values(index):
        a = doc['accessors'][index]
        v = doc['bufferViews'][a['bufferView']]
        fmt = '<' + {5126:'f',5125:'I',5123:'H',5121:'B'}[a['componentType']] * {'VEC3':3,'SCALAR':1}[a['type']]
        offset = v.get('byteOffset',0) + a.get('byteOffset',0)
        return [struct.unpack_from(fmt,buffers[v['buffer']],offset+i*v.get('byteStride',struct.calcsize(fmt))) for i in range(a['count'])]

    black = len(doc['materials'])
    doc['materials'].append({'name':'reference_black_interior','pbrMetallicRoughness':{
        'baseColorFactor':[.012,.012,.012,1.0],'roughnessFactor':.65,'metallicFactor':0.0}})
    raw = bytearray()
    groups = []
    for mesh in doc['meshes']:
        primitives = []
        for prim in mesh['primitives']:
            assert prim.get('mode',4)==4
            positions = values(prim['attributes']['POSITION'])
            indices = [i[0] for i in values(prim['indices'])]
            selected, preserved = [], []
            for i in range(0,len(indices),3):
                tri=indices[i:i+3]
                points=[positions[j] for j in tri]
                c=[sum(p[k] for p in points)/3 for k in range(3)]
                u=[points[1][k]-points[0][k] for k in range(3)]
                v=[points[2][k]-points[0][k] for k in range(3)]
                n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
                # Input visual uses glTF Y-up, front +Z, original authored meters.
                central=max(abs(p[0]) for p in points)<.1801
                back_inside=c[2]<-.149 and n[2]>0
                bottom_inside=c[1]<.05 and n[1]>0
                top_inside=c[1]>.39 and n[1]<0
                inside_sides=abs(c[0])>.45 and c[0]*n[0]<0
                target=selected if whole or central or back_inside or bottom_inside or top_inside or inside_sides else preserved
                target.extend(tri)
            assert selected and (whole or preserved)
            for indices_group, material in ((preserved,prim.get('material',0)),(selected,black)):
                if not indices_group:
                    continue
                offset=len(raw)
                raw.extend(struct.pack('<'+'I'*len(indices_group),*indices_group))
                view=len(doc['bufferViews'])
                doc['bufferViews'].append({'buffer':len(doc['buffers']),'byteOffset':offset,'byteLength':len(raw)-offset,'target':34963})
                accessor=len(doc['accessors'])
                doc['accessors'].append({'bufferView':view,'componentType':5125,'count':len(indices_group),'type':'SCALAR'})
                updated=copy.deepcopy(prim)
                updated.update(indices=accessor,material=material)
                primitives.append(updated)
            # Exactly the original triangle index multiset, only material assignment changes.
            assert sorted(tuple(indices[j:j+3]) for j in range(0,len(indices),3)) == sorted(tuple(g[j:j+3]) for g in (selected,preserved) for j in range(0,len(g),3))
            groups.append({'black_triangles':len(selected)//3,'wood_triangles':len(preserved)//3})
        mesh['primitives']=primitives
    binary=path.stem+'_material_indices.bin'
    (path.parent/binary).write_bytes(raw)
    doc['buffers'].append({'uri':binary,'byteLength':len(raw)})
    path.write_text(json.dumps(doc,indent=2)+'\n')
    return groups


def repair(source, output):
    import numpy as np
    import trimesh
    import xml.etree.ElementTree as ET
    from pydrake.all import RigidTransform, RollPitchYaw
    from scenecode.agent_utils.sdf_generator import YUP_TO_ZUP_TRANSFORM
    source,output=source.resolve(),output.resolve()
    output.mkdir(exist_ok=False,parents=True)
    shutil.copytree(source/'room',output/'room',ignore=shutil.ignore_patterns('house.blend','independent_validation.json'))
    shutil.copy2(source/'recovery_config.yaml',output/'recovery_config.yaml')
    for name in ('integration.json','sofa_independent_review.json'):
        shutil.copy2(source/name,output/name)
    room=output/'room/scene_000/room_living_room'
    asset=room/'generated_assets/furniture/sdf/tv_console_1790964015'
    protected={str(p.relative_to(output)):digest(p) for p in (output/'room').rglob('*') if p.is_file() and (p.suffix in ('.sdf','.obj') or p.name=='house_state.json')}
    report={'kind':'assisted_console_material_recovery','source':str(source),'new_model_calls':0,'physics_rerun_required':False,'geometry_changed':False,'changes':{}}
    for filename,whole in [('base_plinth_visual.gltf',True),('cabinet_body_visual.gltf',False)]:
        report['changes'][filename]=blacken_visual(asset/'visual'/filename,whole)
    # Rebuild the non-articulated preview from the same corrected SDF visuals.
    model=ET.parse(next(asset.glob('*.sdf'))).getroot().find('model')
    def pose(text):
        x,y,z,r,p,yaw=map(float,(text or '0 0 0 0 0 0').split())
        return RigidTransform(RollPitchYaw(r,p,yaw),[x,y,z]).GetAsMatrix4()
    scene=trimesh.Scene()
    for link in model.findall('link'):
        link_pose=pose(link.findtext('pose'))
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            part=trimesh.load(asset/mesh.findtext('uri'),force='scene')
            scale=np.diag([*map(float,mesh.findtext('scale','1 1 1').split()),1.])
            transform=np.linalg.inv(YUP_TO_ZUP_TRANSFORM)@pose(model.findtext('pose'))@link_pose@pose(visual.findtext('pose'))@YUP_TO_ZUP_TRANSFORM@scale
            part.apply_transform(transform)
            scene=trimesh.scene.scene.append_scenes([scene,part],common=None)
    combined=room/'generated_assets/furniture/geometry/tv_console_1790964015/tv_console_combined.gltf'
    old=trimesh.load(combined,force='scene')
    assert np.allclose(scene.bounds,old.bounds,atol=1e-5),(scene.bounds,old.bounds)
    for filename,data in scene.export(file_type='gltf').items():
        (combined.parent/(combined.name if filename=='model.gltf' else filename)).write_bytes(data)
    report['protected_files_unchanged']=all(digest(output/p)==h for p,h in protected.items())
    assert report['protected_files_unchanged']
    report['protected_file_sha256']=protected
    report['state']='ready_for_gpu_export'
    (output/'material_repair.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    repair(a.source,a.output)
