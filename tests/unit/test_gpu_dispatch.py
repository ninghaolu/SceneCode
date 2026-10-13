import json
import pickle
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scenecode.utils import gpu_dispatch as module
from scenecode.utils.gpu_blender_command import needs_gpu


def test_dispatch_result_and_resource_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('SCENECODE_GPU_STAGE_DIR',str(tmp_path))
    monkeypatch.setenv('SLURM_JOB_ID','parent-job')
    monkeypatch.setenv('SBATCH_GRES','gpu:8')
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        assert args[0]=='sbatch'
        assert 'SLURM_JOB_ID' not in kwargs['env']
        assert 'SBATCH_GRES' not in kwargs['env']
        script=Path(args[-1])
        assert '#SBATCH --export=NONE' in script.read_text()
        assert '#SBATCH --time=00:15:00' in script.read_text()
        (script.parent/'result.pkl').write_bytes(pickle.dumps([Path('/shared/render.png')]))
        return SimpleNamespace(stdout='123;cluster\n')
    monkeypatch.setattr(module.subprocess,'run',run)
    assert module.dispatch('some.module','function',(7,),{}) == [Path('/shared/render.png')]
    assert len(calls)==1
    report=json.loads(next(tmp_path.glob('*/dispatch.json')).read_text())
    assert report['state']=='completed'


def test_dispatch_timeout_cancels_its_own_job(tmp_path,monkeypatch):
    monkeypatch.setenv('SCENECODE_GPU_STAGE_DIR',str(tmp_path))
    monkeypatch.setenv('SCENECODE_GPU_WAIT_SECONDS','-1')
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='456\n')
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(TimeoutError):
        module.dispatch('module','function',(),{})
    assert calls[-1]==['scancel','456']


def test_worker_error_is_not_silently_retried(tmp_path,monkeypatch):
    monkeypatch.setenv('SCENECODE_GPU_STAGE_DIR',str(tmp_path))
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[0]=='sbatch':
            (Path(args[-1]).parent/'error.json').write_text('{"traceback":"render failed"}')
        return SimpleNamespace(stdout='789\n')
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='render failed'):
        module.dispatch('module','function',(),{})
    assert [c[0] for c in calls]==['sbatch','scancel']


def test_worker_does_not_recursively_allocate(monkeypatch):
    monkeypatch.setenv('SCENECODE_RENDER_BACKEND','slurm')
    monkeypatch.setenv('SCENECODE_GPU_WORKER','1')
    @module.gpu_stage
    def local(value):
        return value+1
    assert local(1)==2


def test_only_blender_rendering_and_exports_need_gpu():
    assert needs_gpu('render_scene',{})
    assert needs_gpu('export_object_to_obj_with_baked_materials',{})
    assert needs_gpu('execute_code',{'code':'bpy.ops.render.render(write_still=True)'})
    assert not needs_gpu('execute_code',{'code':'bpy.ops.mesh.primitive_cube_add()'})
    assert not needs_gpu('execute_code',{'code':'bpy.ops.wm.save_as_mainfile(filepath="state.blend")'})


def test_dispatch_survives_accounting_timeout_until_result(tmp_path, monkeypatch):
    monkeypatch.setenv('SCENECODE_GPU_STAGE_DIR', str(tmp_path))
    calls = []
    def run(args, **kwargs):
        calls.append(args[0])
        if args[0] == 'sbatch':
            return SimpleNamespace(stdout='123\n')
        assert args[0] == 'sacct'
        raise subprocess.TimeoutExpired(args, 10)
    def publish(_):
        next(tmp_path.iterdir()).joinpath('result.pkl').write_bytes(pickle.dumps('ok'))
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module.time, 'sleep', publish)
    assert module.dispatch('module', 'function', (), {}) == 'ok'
    assert calls == ['sbatch', 'sacct']


def test_accounting_parses_root_cancellation(monkeypatch):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout='CANCELLED by 0|\n'))
    assert module.accounting_state('123') == 'CANCELLED'
