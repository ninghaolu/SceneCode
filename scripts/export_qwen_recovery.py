"""Export a copied, settled recovery checkpoint using a short GPU allocation."""
import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from omegaconf import OmegaConf
from scenecode.agent_utils.house import HouseScene
from scenecode.agent_utils.room import ObjectType
from scripts.validate_qwen_room import validate


def export(source, output):
    source, output = source.resolve(), output.resolve()
    if (output / 'room').exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    record = {'kind': 'recovery_export_not_fresh_baseline', 'source': str(source),
              'state': 'running', 'started_at': time.time(), 'new_model_calls': 0,
              'visual_review': 'Final room review pending; see source records for asset review'}
    path = output / 'recovery.json'
    path.write_text(json.dumps(record, indent=2))
    try:
        shutil.copytree(source / 'room', output / 'room')
        shutil.copy2(source / 'recovery_config.yaml', output / 'recovery_config.yaml')
        for name in ('integration.json', 'sofa_independent_review.json'):
            if (source/name).exists():
                shutil.copy2(source/name,output/name)
        cfg = OmegaConf.to_container(OmegaConf.load(output / 'recovery_config.yaml'), resolve=True)
        house_dir = output / 'room/scene_000'
        state = json.loads((house_dir / 'combined_house_after_furniture/house_state.json').read_text())
        house = HouseScene.from_state_dict(state, house_dir=house_dir)
        house.assemble(cfg=cfg, output_name='combined_house_after_furniture',
                       include_object_types=[ObjectType.FURNITURE])
        report = validate(output / 'room')
        (output / 'room/independent_validation.json').write_text(json.dumps(report, indent=2))
        record.update(state='exported_pending_visual_review' if report['passed'] else 'validation_failed',
                      numeric_validation_passed=report['passed'], errors=report['errors'])
        return report['passed']
    except Exception as exc:
        record.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_at'] = time.time()
        path.write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if export(args.source, args.output) else 1)
