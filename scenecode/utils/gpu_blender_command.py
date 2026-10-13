"""Execute one stateful Blender render/export command in a short GPU job."""
import asyncio
import os
import uuid
from pathlib import Path

from scenecode.utils.gpu_dispatch import dispatch


def needs_gpu(command, params):
    if command in {'render_scene','render_object'} or command.startswith('export_'):
        return True
    code=(params or {}).get('code','')
    return command=='execute_code' and ('bpy.ops.render' in code or 'bpy.ops.object.bake' in code)


async def dispatch_command(client, command, params):
    checkpoint=Path(os.environ['SCENECODE_GPU_STAGE_DIR']).resolve()/'blender_states'/f'{uuid.uuid4().hex}.blend'
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    saved=await client._send_command('execute_code',{'code':f'import bpy\nbpy.ops.wm.save_as_mainfile(filepath={str(checkpoint)!r})'})
    if not saved.success:
        return saved.to_dict()
    result=await asyncio.to_thread(dispatch,__name__,'run_command',(checkpoint,command,params),{})
    restored=await client._send_command('execute_code',{'code':f'import bpy\nbpy.ops.wm.open_mainfile(filepath={str(checkpoint)!r})'})
    if not restored.success:
        raise RuntimeError(f'Failed to restore GPU-rendered Blender state: {restored.error}')
    return result


def run_command(checkpoint, command, params):
    return asyncio.run(_run_command(checkpoint,command,params))


async def _run_command(checkpoint, command, params):
    from scenecode.code_object.agent_framework.blender.backend_manager import BlenderBackendManager
    from scenecode.code_object.agent_framework.blender.mcp_client import BlenderMCPClient
    from scenecode.utils.network_utils import find_available_port
    port=find_available_port(host='127.0.0.1',port_range=(9800,9900))
    backend=BlenderBackendManager('blender',host='127.0.0.1',port=port,
        log_file=str(checkpoint.with_suffix('.worker.log')))
    client=BlenderMCPClient(host='127.0.0.1',port=port,timeout=600)
    try:
        await backend.start()
        if not await client.connect():
            raise RuntimeError('GPU Blender client could not connect')
        loaded=await client.execute_code(f'import bpy\nbpy.ops.wm.open_mainfile(filepath={str(checkpoint)!r})')
        if not loaded['success']:
            raise RuntimeError(str(loaded))
        result=await client._send_command(command,params)
        saved=await client.execute_code(f'import bpy\nbpy.ops.wm.save_as_mainfile(filepath={str(checkpoint)!r})')
        if not saved['success']:
            raise RuntimeError(str(saved))
        return result.to_dict()
    finally:
        await client.disconnect()
        await backend.stop()
