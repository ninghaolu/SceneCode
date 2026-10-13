"""CPU-driven integration checks for all fresh-baseline rendering boundaries."""
import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf
from scenecode.agent_utils.blender import BlenderServer
from scenecode.agent_utils.house import HouseScene
from scenecode.agent_utils.rendering import render_scene_for_agent_observation, save_directive_as_blend
from scenecode.floor_plan_agents.tools.vision_tools import render_floor_plan_directive
from scenecode.code_object.agent_framework.blender.backend_manager import BlenderBackendManager
from scenecode.code_object.agent_framework.blender.mcp_client import BlenderMCPClient
from scenecode.utils.network_utils import find_available_port


async def check_stateful_blender(root):
    port=find_available_port(host='127.0.0.1',port_range=(9800,9900))
    backend=BlenderBackendManager('blender',port=port,log_file=str(root/'cpu_backend.log'))
    client=BlenderMCPClient(host='127.0.0.1',port=port)
    try:
        await backend.start()
        assert await client.connect()
        code="import bpy\nbpy.ops.wm.read_factory_settings(use_empty=True)\nbpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,.5))\nbpy.context.object.name='preserved_cube'"
        assert (await client.execute_code(code))['success']
        before=await client.get_scene_bounds()
        result=await client.render_full_object_v3(str(root/'cube.png'),resolution=(128,128),samples=4,engine='EEVEE')
        assert result['success'],result
        assert (root/'cube.png').is_file()
        after=await client.get_scene_bounds()
        assert before==after,(before,after)
        names=await client.list_scene_object_names()
        assert any(n['name']=='preserved_cube' for n in names)
        await client.execute('export_scene_to_glb',{'output_path':str(root/'cube.glb')})
        assert (root/'cube.glb').is_file()
        return {'passed':True,'bounds_unchanged':True}
    finally:
        await client.disconnect()
        await backend.stop()


def main(root):
    root.mkdir(parents=True,exist_ok=True)
    os.environ['SCENECODE_RENDER_BACKEND']='slurm'
    os.environ['SCENECODE_GPU_STAGE_DIR']=str(root/'gpu_tasks')
    source=Path('/scratch/nl2752/SceneCode/outputs/qwen_recovery_export_19105378')
    house_dir=source/'room/scene_000'
    state=json.loads((house_dir/'combined_house_after_furniture/house_state.json').read_text())
    house=HouseScene.from_state_dict(state,house_dir=house_dir)
    report={'passed':False,'checks':{},'started_at':time.time(),'new_model_calls':0}
    path=root/'smoke.json'
    def record(name,value):
        report['checks'][name]=value
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(name,value,flush=True)
    server=BlenderServer()
    server.start()
    assert server._server_process is None,'CPU caller unexpectedly started render server'
    try:
        record('stateful_blender',asyncio.run(check_stateful_blender(root)))
        render_floor_plan_directive(house_dir/'combined_house_after_furniture/house.dmd.yaml',house_dir,server,root/'floor.png',256)
        assert (root/'floor.png').is_file()
        record('floor_observation',{'passed':True})
        cfg=OmegaConf.load(source/'recovery_config.yaml').furniture_agent.rendering
        for key in ('top_view_width','top_view_height','side_view_width','side_view_height'):
            cfg[key]=256
        images=render_scene_for_agent_observation(house.rooms['living_room'],cfg,server,override_side_view_count=1,include_vertical_views=False,taa_samples=4)
        assert images and all(p.is_file() for p in images)
        record('room_observation',{'passed':True,'images':list(map(str,images))})
        images=server.render_multiview_for_analysis(root/'cube.glb',root/'asset_views',20,num_side_views=1,include_vertical_views=False,width=128,height=128,taa_samples=4)
        assert images and all(p.is_file() for p in images)
        record('asset_observation',{'passed':True})
        save_directive_as_blend(house_dir/'combined_house_after_furniture/house.dmd.yaml',root/'house.blend',scene_dir=house_dir)
        assert (root/'house.blend').stat().st_size>100000
        record('final_export',{'passed':True})
        report['passed']=True
    finally:
        server.stop()
        report['finished_at']=time.time()
        repo=Path(__file__).resolve().parents[1]
        names=['scenecode/utils/gpu_dispatch.py','scenecode/utils/gpu_blender_command.py','scenecode/agent_utils/blender/server_manager.py','scenecode/agent_utils/rendering.py','scenecode/floor_plan_agents/tools/vision_tools.py','scenecode/code_object/agent_framework/blender/mcp_client.py']
        report['source_sha256']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in names}
        path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    main(parser.parse_args().output.resolve())
