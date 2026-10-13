import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo = Path('/scratch/nl2752/SceneCode')
output = repo / 'outputs/qwen_resume_20261005'
record_path = output / 'submission.json'
if record_path.exists():
    raise SystemExit('Already submitted; inspect submission.json instead of duplicating')
assert (output / 'resume_config.yaml').is_file()
env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'SBATCH_'))}
def submit(args):
    result = subprocess.run(['sbatch', '--parsable', *args], cwd=repo, env=env,
                            capture_output=True, text=True, check=True, timeout=60)
    job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected submission response: {result.stdout}')
    return job
smoke = submit(['slurm/smoke_gpu_dispatch.sbatch'])
report = repo / f'outputs/gpu_split_smoke_{smoke}/smoke.json'
record = {'submitted_at': datetime.now(timezone.utc).isoformat(),
          'source_job': '19192619', 'smoke_job': smoke, 'smoke_report': str(report),
          'output': str(output), 'gpu_job_time_limit': '00:15:00',
          'parent_time_limit': '06:00:00', 'render_backend': 'slurm',
          'reused_assets': ['bookshelf_0', 'coffee_table_0', 'tv_console_cabinet_0'],
          'tests': '13 targeted tests passed; shell/Python syntax and diff whitespace passed',
          'state': 'smoke_submitted'}
record_path.write_text(json.dumps(record, indent=2)+'\n')
resume = submit([f'--dependency=afterok:{smoke}', '--kill-on-invalid-dep=yes',
                 'slurm/resume_living_room_qwen.sbatch', str(repo/'outputs/qwen_19192619'),
                 str(output), str(report)])
record.update(resume_job=resume, state='submitted_after_smoke')
record_path.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
