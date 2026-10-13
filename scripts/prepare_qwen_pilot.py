"""Write isolated, resolved Code_Object configs for every profile."""
import argparse
import os
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[1]
for filename in ("config.yaml", "config_simplemanip.yaml", "config_structmanip.yaml", "config_WallArt.yaml"):
    config = yaml.safe_load((root / "scenecode/code_object/agent_framework" / filename).read_text())
    for role in config["model"].values():
        role.update(provider="openai", model_name=os.environ["SCENECODE_QWEN_MODEL"],
                    api_base=os.environ["OPENAI_API_BASE"], max_tokens=int(os.getenv("SCENECODE_MAX_TOKENS", "16384")))
    config["agents"]["planner_checker"]["max_retry"] = 3
    config["agents"]["part_constructor"]["max_retry"] = 3
    config["pipeline"]["require_valid_plan"] = True
    config["pipeline"]["validate_part_positions"] = True
    (args.output / filename).write_text(yaml.safe_dump(config, sort_keys=False))
