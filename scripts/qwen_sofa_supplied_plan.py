"""Bounded fresh Blender implementation from a frozen, reviewed sofa plan."""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from scenecode.utils.openai import create_openai_client, extract_openai_usage_stats
from scripts.qwen_sofa_repair import image_content, render, validate_patch
from scripts.sofa_supplied_plan import GROUPS, check_geometry


def run(plan_dir, output, attempts):
    plan_dir, output = plan_dir.resolve(), output.resolve()
    review = json.loads((plan_dir/'plan_review.json').read_text())
    plan_bytes = (plan_dir/'plan.json').read_bytes()
    digest = hashlib.sha256(plan_bytes).hexdigest()
    if review.get('passed') is not True or review.get('plan_sha256') != digest:
        raise ValueError('Supplied plan must have a passing review of this exact version')
    plan = json.loads(plan_bytes)
    if hashlib.sha256((plan_dir/'reference.png').read_bytes()).hexdigest() != plan['reference_sha256']:
        raise ValueError('Reference differs from the frozen plan')
    if (output/'plan.json').exists():
        raise FileExistsError('Use a fresh experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('plan.json', 'plan_review.json', 'reference.png'):
        shutil.copy2(plan_dir/name, output/name)
    model = os.environ.get('SCENECODE_QWEN_MODEL', 'qwen3.7-plus')
    record = dict(kind='assisted_supplied_plan_fresh_code_diagnostic', model=model,
                  plan_source=str(plan_dir), plan_sha256=digest, max_attempts=attempts,
                  state='running', attempts=[], autonomous_baseline=False,
                  interventions=['Agent-authored explicit dimensions, instances and support assumptions',
                                 'Deterministic blockout reviewed before generation',
                                 'Fresh full Blender code; no old code, construction library or blockout code supplied'],
                  acceptance=plan['acceptance'])
    result_path = output/'supplied_plan_result.json'
    result_path.write_text(json.dumps(record, indent=2))
    client = create_openai_client().with_options(timeout=300, max_retries=0)
    started = time.monotonic()
    feedback = None
    previous_files = None
    try:
        for index in range(1, attempts+1):
            directory = output/f'attempt_{index}'
            directory.mkdir()
            item = dict(attempt=index, passed=False)
            record['attempts'].append(item)
            prompt = (
                'Implement this complete, pre-reviewed sofa plan as Blender 4.4 Python code. '
                'The reference image supplies appearance; the frozen plan supplies authoritative metric bounds. '
                'This tests your implementation GIVEN a plan, not your ability to infer the plan. '
                'Return JSON {"explanation":string,"files":{filename:full_source}} with exactly five files: '
                'base_frame.py, legs.py, armrests.py, seat_cushions.py, back_cushions.py. '
                'Each file runs independently using runpy as __main__, in that order in one persistent scene. '
                'Each must create ONLY its listed group components. Use unique object names from the plan. '
                'Use only bpy, math, mathutils imports. No file access, other imports, dynamic execution, '
                'scene-wide deletion, lighting, cameras, export, or modifying previously created objects. '
                'Code must create geometry when executed, not just define functions. '
                'Use explicit instantiation for left/right and repeated cushions to avoid mirror-origin mistakes. '
                'Applied/evaluated geometry, not just raw cube parameters, must have the planned dimensions and centers. '
                'Use scale-applied cubes and controlled bevels with 5 segments, rather than subdivision shrinkage. '
                'Foot shape is a 32-vertex cone, bottom radius .025 m, top radius .034 m, height .125 m. '
                'No additional disconnected piping, seams or decoration. Rounded boxes have the exact bevels specified. '
                'All fabric should share a light-gray rough material, wood feet a natural brown material. '
                'The hidden rear support and intentional small contact overlaps are approved assumptions. '
                'Do not change proportions, normalization, floor, axes or these assumptions to resemble a different view. '
                'Keep code concise; a small local helper per file is fine. No construction helper/library is provided.\n'
                + json.dumps(plan, indent=2))
            if feedback is not None:
                prompt += '\nPrevious attempt and measured/visual feedback:\n' + json.dumps(
                    dict(files=previous_files, feedback=feedback))
            (directory/'prompt.txt').write_text(prompt)
            response = client.chat.completions.create(model=model, response_format={'type':'json_object'},
                messages=[{'role':'user', 'content':[{'type':'text', 'text':prompt}, image_content(output/'reference.png')]}])
            item['generation_usage'] = extract_openai_usage_stats(response, 'chat_completions')
            raw = response.choices[0].message.content
            (directory/'response.json').write_text(raw or '')
            try:
                proposal = json.loads(raw)
                files = proposal['files']
                validate_patch(files, {group+'.py' for group in GROUPS})
                previous_files = files
                code = directory/'code'
                code.mkdir()
                for name, source in files.items():
                    (code/name).write_text(source)
                measurements = render(code, directory, framing_scale=2)
                checks = check_geometry(plan, measurements)
                (directory/'checks.json').write_text(json.dumps(checks, indent=2))
                item['checks'] = checks
                # A separate call critiques images without receiving numeric pass/fail.
                instruction = (
                    'Compare reference image 1 with ONE generated sofa: image 2 front, '
                    '3 front three-quarter, 4 rear three-quarter. Judge coarse structural fidelity. '
                    'Require two opposite padded arms, three broad rectangular back cushions, three '
                    'seat cushions fitting between arms, short wooden feet, low wide proportions '
                    'and plausible support. Reject missing or misplaced components, obvious floating, '
                    'gross intersections, severe shading artifacts, dome-shaped backs or major proportion errors. '
                    'Occlusion in the rear view is expected. This supplied-plan diagnostic explicitly accepts '
                    'smooth simple fabric, no seams/wrinkles, axis-aligned cushions, straight tapered feet '
                    'and assumed rear backing; those limitations alone are not failures. '
                    'Do not claim photorealism or exact reconstruction. Return JSON '
                    '{"acceptable":boolean,"defects":[string],"limitations":[string],"reason":string}.')
                (directory/'critique_prompt.txt').write_text(instruction)
                critique_response = client.chat.completions.create(model=model,
                    response_format={'type':'json_object'}, messages=[{'role':'user','content':[
                        {'type':'text','text':instruction}, image_content(output/'reference.png'),
                        *[image_content(directory/f'{view}.png') for view in ('front','three_quarter','back')]]}])
                item['critique_usage'] = extract_openai_usage_stats(critique_response, 'chat_completions')
                critique_raw = critique_response.choices[0].message.content
                (directory/'critique_response.json').write_text(critique_raw or '')
                critique = json.loads(critique_raw)
                feedback = dict(numeric=checks, visual=critique)
                item.update(visual_critique=critique, passed=checks['passed'] and critique.get('acceptable') is True)
            except Exception as exc:
                feedback = dict(error=f'{type(exc).__name__}: {exc}')
                log = directory/'blender.log'
                if log.exists():
                    feedback['blender_log_tail'] = log.read_text()[-8000:]
                item['failure'] = feedback
            result_path.write_text(json.dumps(record, indent=2))
            if item['passed']:
                record.update(state='candidate_ready_for_independent_review', selected_attempt=f'attempt_{index}')
                return True
        record['state'] = 'implementation_budget_exhausted'
        return False
    except Exception as exc:
        record.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds'] = time.monotonic()-started
        result_path.write_text(json.dumps(record, indent=2))
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempts', type=int, choices=(1,2), default=2)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.plan, args.output, args.attempts) else 1)
