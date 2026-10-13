"""Run a prepared fresh room configuration through the existing experiment."""
import argparse
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if cfg.experiment.pipeline.start_stage != 'floor_plan':
        raise ValueError('Fresh run must begin at floor_plan')
    if cfg.experiment.pipeline.get('resume_from_path') or cfg.furniture_agent.asset_manager.get('resume_registry_path'):
        raise ValueError('Fresh run must not reuse a floor plan or furniture registry')
    expected = os.environ['SCENECODE_REQUIRED_MODEL']
    for name in ('floor_plan_agent', 'furniture_agent'):
        if cfg[name].openai.model != expected or cfg[name].session_memory.summarization_model != 'agent':
            raise ValueError(f'Unexpected model settings in {name}')
    if Path(cfg.experiment.output_dir).exists():
        raise FileExistsError(f'Refusing to overwrite room output: {cfg.experiment.output_dir}')
    if os.environ.get('SCENECODE_RENDER_BACKEND') != 'slurm':
        raise ValueError('This launch requires short GPU dispatch')
    if args.check_only:
        print('Fresh room configuration checked; no generation performed')
        return
    import bpy  # Import before Drake/other native libraries, as in the resume.
    from scenecode.experiments import build_experiment
    from scenecode.utils.omegaconf import register_resolvers
    register_resolvers()
    OmegaConf.resolve(cfg)
    output = Path(cfg.experiment.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / 'resolved_config.yaml')
    experiment = build_experiment(cfg)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    main()
