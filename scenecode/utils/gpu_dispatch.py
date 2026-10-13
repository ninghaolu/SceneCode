"""Run rendering work in bounded Slurm allocations, using shared-filesystem IPC.

Only trusted local Python payloads are accepted. Workers have no API credentials;
the CPU caller retains model orchestration and blocks until its render completes.
"""
import functools
import importlib
import json
import os
import pickle
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path


def enabled():
    return os.getenv('SCENECODE_RENDER_BACKEND') == 'slurm' and os.getenv('SCENECODE_GPU_WORKER') != '1'


def accounting_state(job):
    """Accounting remains usable when the live queue controller is unavailable."""
    try:
        result = subprocess.run(
            ['sacct', '-n', '-X', '-j', job, '--format=State', '--parsable2'],
            text=True, capture_output=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    return lines[0].split('|')[0].split()[0] if lines else None


def dispatch(module, name, args, kwargs):
    root = Path(os.environ['SCENECODE_GPU_STAGE_DIR']).resolve()
    task = root / uuid.uuid4().hex
    task.mkdir(parents=True)
    task.chmod(0o700)
    with (task/'input.pkl').open('wb') as f:
        pickle.dump((module,name,args,kwargs),f)
    repo = Path(__file__).resolve().parents[2]
    script = task/'worker.sbatch'
    script.write_text('\n'.join([
        '#!/bin/bash', '#SBATCH --job-name=scenecode-render',
        '#SBATCH --nodes=1', '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=8',
        '#SBATCH --mem=32G', '#SBATCH --time=00:15:00',
        '#SBATCH --constraint=l40s', '#SBATCH --gres=gpu:1',
        '#SBATCH --account=torch_pr_56_tandon_advanced', '#SBATCH --export=NONE',
        'set -eo pipefail', 'source /scratch/nl2752/scenecode_setup/activate.sh',
        'cd '+shlex.quote(str(repo)), 'export PYTHONPATH='+shlex.quote(str(repo)),
        'export SCENECODE_GPU_WORKER=1', 'export OMP_NUM_THREADS=8',
        'export SCENECODE_BLENDER_GLOBAL_LOCK='+shlex.quote(str(task/'blender.lock')),
        'export TMPDIR='+shlex.quote(str(task/'tmp')), 'mkdir -p "$TMPDIR"',
        shlex.quote(sys.executable)+' -m scenecode.utils.gpu_dispatch '+shlex.quote(str(task)), '']))
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','SBATCH_'))}
    submission=subprocess.run(['sbatch','--parsable','--output',str(task/'job.out'),'--error',str(task/'job.err'),str(script)],env=env,check=True,capture_output=True,text=True)
    job=submission.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected sbatch result: {submission.stdout!r}')
    record={'job_id':job,'operation':module+'.'+name,'state':'submitted','submitted_at':time.time()}
    path=task/'dispatch.json'
    path.write_text(json.dumps(record,indent=2))
    deadline=time.monotonic()+float(os.getenv('SCENECODE_GPU_WAIT_SECONDS','3600'))
    completed=False
    terminal_since=None
    next_status_check=0.0
    last_state=None
    try:
        while time.monotonic()<deadline:
            result=task/'result.pkl'
            error=task/'error.json'
            if error.exists():
                raise RuntimeError(f'GPU job {job} failed: {error.read_text()}')
            if result.exists():
                with result.open('rb') as f:
                    value=pickle.load(f)
                completed=True
                record.update(state='completed',finished_at=time.time())
                return value
            if time.monotonic() >= next_status_check:
                last_state=accounting_state(job)
                next_status_check=time.monotonic()+30
                if last_state in {'FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','COMPLETED'}:
                    # NFS negative directory/attribute caches can outlive the
                    # allocation by tens of seconds. Give atomic result publication
                    # a bounded visibility grace period before declaring failure.
                    terminal_since=terminal_since or time.monotonic()
            if terminal_since is not None and time.monotonic()-terminal_since>120:
                raise RuntimeError(f'GPU job {job} ended {last_state} without a result; see {task}')
            time.sleep(3)
        raise TimeoutError(f'GPU job {job} exceeded queue/execution wait budget; see {task}')
    finally:
        if not completed:
            subprocess.run(['scancel',job],capture_output=True,text=True)
            record.update(state='failed_or_cancelled',finished_at=time.time())
        path.write_text(json.dumps(record,indent=2))


def gpu_stage(func):
    @functools.wraps(func)
    def wrapper(*args,**kwargs):
        if enabled():
            return dispatch(func.__module__,func.__qualname__,args,kwargs)
        return func(*args,**kwargs)
    return wrapper


def worker(task):
    import traceback
    os.environ['SCENECODE_GPU_WORKER']='1'
    servers=[]
    try:
        with (task/'input.pkl').open('rb') as f:
            module,name,args,kwargs=pickle.load(f)
        from scenecode.agent_utils.blender import BlenderServer
        def start_servers(value):
            if isinstance(value,BlenderServer) and value not in servers:
                value._running=False
                value._gpu_id=None
                value.start()
                servers.append(value)
                value.wait_until_ready()
            elif isinstance(value,(tuple,list)):
                for item in value:
                    start_servers(item)
            elif isinstance(value,dict):
                for item in value.values():
                    start_servers(item)
        start_servers((args,kwargs))
        function=importlib.import_module(module)
        for part in name.split('.'):
            function=getattr(function,part)
        value=function(*args,**kwargs)
        temporary=task/'result.tmp'
        with temporary.open('wb') as f:
            pickle.dump(value,f)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(task/'result.pkl')
    except BaseException:
        temporary=task/'error.tmp'
        temporary.write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))
        temporary.replace(task/'error.json')
        raise
    finally:
        for server in servers:
            server.stop()


if __name__=='__main__':
    worker(Path(sys.argv[1]).resolve())
