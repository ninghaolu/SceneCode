"""Resume furniture generation using a copied registry and the saved floor plan."""
import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def copy_assets(source, output):
    source = source.resolve()
    original = source / 'room/scene_000/room_living_room/generated_assets/furniture'
    target = output.resolve() / 'reused_assets/furniture'
    before = digest_tree(original)
    registry = json.loads((original / 'asset_registry.json').read_text())
    if not registry:
        raise ValueError('Source registry is empty')
    for asset in registry.values():
        for key in ('geometry_path', 'sdf_path', 'image_path'):
            path = Path(asset[key]).resolve()
            if not path.is_relative_to(original) or not path.is_file():
                raise ValueError(f'Missing or external {key}: {path}')
    shutil.copytree(original, target)
    for path in target.rglob('*'):
        if path.is_file() and path.suffix.lower() in {
            '.json', '.yaml', '.yml', '.sdf', '.urdf', '.gltf', '.obj', '.mtl', '.py', '.txt'
        }:
            text = path.read_text()
            remapped = text.replace(str(original), str(target))
            if remapped != text:
                path.write_text(remapped)
    if digest_tree(original) != before:
        raise RuntimeError('Source assets changed while copying')
    manifest = {
        'kind': 'resumed_generation_not_fresh_baseline',
        'source_run': str(source), 'created_at': datetime.now(timezone.utc).isoformat(),
        'reused_asset_ids': list(registry), 'missing_asset': 'sofa',
        'source_asset_sha256': before, 'source_assets_unchanged': True,
        'rendering': 'short GPU jobs; no CPU rendering',
        'agent_sessions': 'new conversations with explicit saved-asset reuse instruction',
        'visual_review': 'pending', 'verified_correct_result': False,
    }
    (output / 'resume.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return target / 'asset_registry.json'


def prepare(source, output):
    from omegaconf import OmegaConf
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'room').exists() or (output / 'resume_config.yaml').exists():
        raise FileExistsError(f'Refusing to overwrite resumed run: {output}')
    registry = copy_assets(source, output)
    shutil.copytree(source / 'configs', output / 'configs')
    for path in (output / 'configs').glob('*.yaml'):
        path.write_text(path.read_text().replace(str(source), str(output)))
    cfg = OmegaConf.load(source / 'room/resolved_config.yaml')
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg).replace(str(source / 'configs'), str(output / 'configs')))
    cfg.name = output.name
    cfg.experiment.output_dir = str(output / 'room')
    cfg.experiment.pipeline.start_stage = 'furniture'
    cfg.experiment.pipeline.stop_stage = 'furniture'
    cfg.experiment.pipeline.resume_from_path = str(source / 'room')
    cfg.furniture_agent.asset_manager.resume_registry_path = str(registry)
    OmegaConf.save(cfg, output / 'resume_config.yaml')
    print(f'Prepared resumed run: {output}', flush=True)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.prepared:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(output / 'resume_config.yaml')
        record = json.loads((output / 'resume.json').read_text())
        if record['source_run'] != str(source):
            raise ValueError('Prepared source mismatch')
        if (output / 'room').exists():
            raise FileExistsError('Prepared run has already started')
    else:
        cfg = prepare(source, output)
    if args.prepare_only:
        return
    import os
    if os.environ.get('SCENECODE_RENDER_BACKEND') != 'slurm':
        raise RuntimeError('This resumed run requires short GPU dispatch')
    import bpy  # Import before Drake/other native libraries.
    from scenecode.experiments import build_experiment
    from scenecode.utils.omegaconf import register_resolvers
    register_resolvers()
    experiment = build_experiment(cfg)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    main()
