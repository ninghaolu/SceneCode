"""Summarize recorded usage without inventing provider billing."""
import argparse
import collections
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--exit-code", type=int)
args = parser.parse_args()
totals = collections.Counter()
models = collections.Counter()
statuses = collections.Counter()
components = collections.defaultdict(collections.Counter)
for path in (args.root / "api").glob("*.json"):
    record = json.loads(path.read_text())
    response = record.get("response", {})
    usage = response.get("usage") or {}
    models[record["requested_model"]] += 1
    statuses[str(record.get("status", "unfinished_or_transport_error"))] += 1
    totals["calls"] += 1
    totals["api_seconds_sum_including_parallel_calls"] += record.get("api_seconds", 0)
    totals["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    totals["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    details = usage.get("input_tokens_details", usage.get("prompt_tokens_details")) or {}
    totals["cached_input_tokens"] += details.get("cached_tokens", 0) or 0
    details = usage.get("output_tokens_details", usage.get("completion_tokens_details")) or {}
    totals["reasoning_tokens"] += details.get("reasoning_tokens", 0) or 0
    component_name = record.get("component", "unknown")
    if component_name == "outer_agent":
        for frame in record.get("call_stack", []):
            for stage in ("floor_plan", "furniture"):
                if f"/{stage}_agents/" in frame:
                    role = "critic" if "critique" in frame else "designer" if "design" in frame else "planner"
                    component_name = f"{stage}_{role}"
    component = components[component_name]
    component["calls"] += 1
    component["api_seconds_sum_including_parallel_calls"] += record.get("api_seconds", 0)
    component["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    component["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    component["reasoning_tokens"] += details.get("reasoning_tokens", 0) or 0
report = {"exit_code": args.exit_code, "requested_models": dict(models),
          "http_statuses": dict(statuses), "usage": dict(totals),
          "components": {name: dict(counts) for name, counts in components.items()},
          "actual_billing": None, "billing_note": "Unavailable; no dollar cost inferred from tokens or timing."}
for name, path in {"asset_validation": args.root / "asset_pilot/validation.json",
                   "room_validation": args.root / "room/independent_validation.json"}.items():
    if name == "asset_validation" and not path.exists():
        path = args.root / "asset_validation.json"
    report[name] = json.loads(path.read_text()) if path.exists() else None
report["room_validated"] = bool((report.get("room_validation") or {}).get("passed"))
report["asset_pilot_passed"] = bool((report.get("asset_validation") or {}).get("passed"))
visual_review = args.root / "visual_review.json"
report["visual_review"] = json.loads(visual_review.read_text()) if visual_review.exists() else None
termination = args.root / "termination.json"
report["termination"] = json.loads(termination.read_text()) if termination.exists() else None
report["verified_correct_result"] = bool(
    report["room_validated"] and (report["visual_review"] or {}).get("passed")
    and args.exit_code == 0)
(args.root / "run_summary.json").write_text(json.dumps(report, indent=2))
print("Run summary:", args.root / "run_summary.json")
