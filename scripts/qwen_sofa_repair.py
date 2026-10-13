"""Bounded Qwen local-code repair of one saved sofa; never claims room success."""
import argparse
import ast
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from scenecode.utils.openai import create_openai_client, encode_image_to_base64, extract_openai_usage_stats
from scripts.sofa_repair_checks import check_sofa

REPAIR_FILES = {'armrests.py', 'seat_cushions.py', 'back_cushions.py'}
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/qwen_19051280/room/scene_000/room_living_room/generated_assets/furniture/code_object/sofa_1790964594912'


def validate_patch(files, allowed_files=None):
    if set(files) != (REPAIR_FILES if allowed_files is None else allowed_files):
        raise ValueError('Response must provide exactly the permitted part files')
    for name, code in files.items():
        if not isinstance(code, str) or len(code) > 60000:
            raise ValueError(f'Invalid source for {name}')
        tree = ast.parse(code, filename=name)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module or '']
                if any(x.split('.')[0] not in {'bpy', 'math', 'mathutils'} for x in modules):
                    raise ValueError(f'Unexpected import in {name}')
            if isinstance(node, ast.Name) and node.id in {'open', 'exec', 'eval', 'compile', '__import__'}:
                raise ValueError(f'Unexpected dynamic/file operation in {name}')
        if 'bpy.ops.wm.' in code or 'bpy.data.libraries' in code:
            raise ValueError('File operations belong to the harness, not generated code')
        compile(code, name, 'exec')


def render(code_dir, output, framing_scale=1):
    output.mkdir(parents=True, exist_ok=True)
    child_env = {k:v for k,v in os.environ.items() if not any(word in k.upper() for word in ('API_KEY','TOKEN','SECRET'))}
    child_env['CUDA_VISIBLE_DEVICES'] = ''
    with (output / 'blender.log').open('w') as log:
        subprocess.run(['blender','-b','-t',os.getenv('SLURM_CPUS_PER_TASK','8'),
                        '--python-exit-code','1','--python',str(ROOT/'scripts/sofa_repair_blender.py'),
                        '--','--code',str(code_dir),'--output',str(output),
                        '--framing-scale',str(framing_scale)],
                       stdout=log, stderr=subprocess.STDOUT, env=child_env, check=True, timeout=600)
    return json.loads((output / 'measurements.json').read_text())


def image_content(path):
    return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+encode_image_to_base64(path)}}


def main(output, attempts, start_code=None, proportions_only=False):
    output = output.resolve()
    if (output / 'baseline').exists():
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'slurm/qwen_saved_images.json').read_text())
    reference = Path(next(e['path'] for e in manifest if 'sofa' in e['aliases']))
    baseline_code = output / 'baseline/code'
    shutil.copytree(start_code or SOURCE / f'code/parts_{SOURCE.name}', baseline_code)
    shutil.copy2(reference, output/'reference.png')
    shutil.copy2(SOURCE/'ObjectPlan.json',output/'original_plan.json')
    start = time.monotonic()
    original = render(baseline_code, output/'baseline')
    previous = check_sofa(original)
    (output/'baseline/checks.json').write_text(json.dumps(previous,indent=2))
    if previous['passed']:
        raise RuntimeError('Invalid diagnostic: known defective control was not rejected')
    model = os.environ.get('SCENECODE_QWEN_MODEL','qwen3.7-plus')
    client = create_openai_client().with_options(timeout=180, max_retries=1)
    current_code, current_measurements, current_render = baseline_code, original, output/'baseline'
    record = {'kind':'assisted_local_code_repair_diagnostic', 'model':model, 'source':str(SOURCE),
              'starting_code':str(start_code) if start_code else str(SOURCE / f'code/parts_{SOURCE.name}'),
              'proportions_only':proportions_only,
              'max_attempts':attempts,'attempts':[], 'state':'running',
              'interventions':['Human-identified faulty parts and reference-specific measurements',
                               'Freeze base/legs; repair only armrests/seat/back code',
                               'Standalone Blender CPU render harness; no fresh end-to-end generation'],
              'acceptance':'Independent geometry plus a separate model critique; human visual review still required'}
    path = output/'repair_result.json'
    path.write_text(json.dumps(record,indent=2))
    try:
        for attempt in range(1, attempts+1):
            directory = output/f'attempt_{attempt}'
            directory.mkdir()
            prompt = (
                'Repair the existing Blender sofa code to match reference image 1. Images 2-4 are the current sofa. '
                'This is LOCAL REPAIR: preserve the base_frame and legs code and geometry exactly. '
                'Return JSON {"diagnosis":string,"files":{"armrests.py":full Python source,"seat_cushions.py":full Python source,"back_cushions.py":full Python source}}. '
                'Each file must define its existing create_* function AND invoke it under if __name__ == "__main__". '
                'Use only bpy, math, mathutils. No file access or scene-wide deletion. Do not change other part objects. '
                'Keep the normalized 0.96-wide, about 0.42-deep footprint; the fixed base is centered (0,0,0.15), dimensions (.96,.42,.12). '
                'Z is up, front is -Y. Dimensions are generator units; room scaling occurs later. '
                'The reference has TWO full padded side arms at opposite X sides, THREE side-by-side seat cushions, '
                'THREE broad rounded RECTANGULAR back cushions rising visibly above the seats. Not small spheres/headrests. '
                'Make the cushions fit between the arms and touch supporting structure. Avoid substantial interpenetration. '
                'Known original defects: mirror was baked while arm geometry was at the origin, yielding two overlapping arms on the right; '
                'back subdivision shrank/rounded the shape; seats occlude much of the back and overlap arms. '
                'Choose suitable cushion dimensions/positions from reference rather than preserving those errors. '
                'At least 0.17 units of back height should be exposed above seat top; each back width>=.21, height>=.22. '
                'Prefer controlled bevels over unsupported subdivision that shrinks boxes. '
                'Keep the light-gray fabric appearance; export/lighting is handled outside your code.\n'
                + json.dumps({'current_measurements':current_measurements,'feedback':previous,
                              'current_code':{p.name:p.read_text() for p in current_code.glob('*.py')}},indent=2))
            if proportions_only:
                prompt += ('\nThis is a separate PROPORTION-ONLY follow-up to the failed two-attempt trial. '
                           'The current code already creates the right instance counts and X positions. Preserve '
                           'that working creation/array/centering logic. Change only size and position constants '
                           'in the three existing files, retaining the functions and material code. '
                           'The current back is too tall. Overall height/width must be <=0.55, and each back '
                           'width/height must be >=0.9. Lower the overly thick seat cushions if necessary so '
                           'back height exposed above seats remains >=0.17 while shortening the backs. '
                           'Keep all support contact, symmetry, and cushion alignment; avoid tall armrests. '
                           'Judge the geometry in the images, not code comments that may be stale.')
            (directory/'repair_prompt.txt').write_text(prompt)
            response = client.chat.completions.create(model=model,response_format={'type':'json_object'},messages=[
                {'role':'user','content':[{'type':'text','text':prompt},image_content(reference),
                 *[image_content(current_render/f'{v}.png') for v in ('front','three_quarter','back')]]}])
            raw = response.choices[0].message.content
            (directory/'response.json').write_text(raw)
            proposed = json.loads(raw)
            validate_patch(proposed['files'])
            code_dir = directory/'code'
            shutil.copytree(baseline_code,code_dir)
            for name, code in proposed['files'].items():
                (code_dir/name).write_text(code)
            item = {'attempt':attempt,'repair_usage':extract_openai_usage_stats(response,'chat_completions'),
                    'diagnosis':proposed['diagnosis']}
            record['attempts'].append(item)
            try:
                measurements = render(code_dir,directory)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                previous={'execution_error':str(exc),'log':(directory/'blender.log').read_text()[-12000:]}
                item.update(passed=False,feedback=previous)
                path.write_text(json.dumps(record,indent=2))
                continue
            checks = check_sofa(measurements,original)
            (directory/'checks.json').write_text(json.dumps(checks,indent=2))
            instruction = ('Independently compare reference image 1 and generated sofa images: '
                           '2=FRONT, 3=FRONT THREE-QUARTER, 4=REAR THREE-QUARTER. '
                           'These are views of ONE object. Parts legitimately occluded in the rear view '
                           'are not missing if visible in the front views. '
                           'Ignore texture detail and lighting but reject missing/overlapping arms, dome-like backs, '
                           'major proportion errors, seat/back intersections, floating pieces or failure to reproduce '
                           'three rectangular seat and three broad rectangular back cushions. '
                           'Both padded side arms must be present and seats must fit between them. '
                           'Do not accept simply because numeric checks pass. '
                           'Return JSON {"acceptable":boolean,"defects":[string],"reason":string}.\n'+json.dumps(checks))
            critique_response = client.chat.completions.create(model=model,response_format={'type':'json_object'},messages=[
                {'role':'user','content':[{'type':'text','text':instruction},image_content(reference),
                 *[image_content(directory/f'{v}.png') for v in ('front','three_quarter','back')]]}])
            critique = json.loads(critique_response.choices[0].message.content)
            item.update(checks=checks,visual_critique=critique,
                        critique_usage=extract_openai_usage_stats(critique_response,'chat_completions'),
                        passed=checks['passed'] and critique.get('acceptable') is True)
            previous={'numeric':checks,'visual':critique}
            current_code,current_measurements,current_render=code_dir,measurements,directory
            path.write_text(json.dumps(record,indent=2))
            if item['passed']:
                record.update(state='candidate_ready_for_independent_review',selected_attempt=attempt)
                return True
        record['state']='repair_budget_exhausted'
        return False
    except Exception as exc:
        record.update(state='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds']=time.monotonic()-start
        path.write_text(json.dumps(record,indent=2))
        client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--attempts',type=int,choices=(1,2),default=2)
    parser.add_argument('--start-code',type=Path)
    parser.add_argument('--proportions-only',action='store_true')
    args=parser.parse_args()
    raise SystemExit(0 if main(args.output,args.attempts,args.start_code,args.proportions_only) else 1)
