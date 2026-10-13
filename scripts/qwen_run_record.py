"""Persist a credential-free source snapshot and machine-readable job status."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--exit-code", type=int)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    status_path = args.root / "status.json"
    now = datetime.now(timezone.utc).isoformat()
    if args.snapshot:
        destination = args.root / "source_snapshot"
        destination.mkdir(exist_ok=False)
        files = subprocess.check_output(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=repo).decode().split("\0")
        allowed_dirs = {"scenecode", "configs", "scripts", "slurm", "tests"}
        allowed_root_files = {"main.py", "pyproject.toml", "uv.lock", "README.md"}
        hashes = {}
        for name in sorted(set(files) - {""}):
            relative = Path(name)
            if relative.parts[0] not in allowed_dirs and name not in allowed_root_files:
                continue
            source = repo / relative
            if not source.is_file() or source.is_symlink():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        (destination / "implementation.diff").write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo))
        manifest = {"created_at": now, "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip(), "sha256": hashes,
            "settings": {key: os.environ.get(key) for key in (
                "SCENECODE_QWEN_MODEL", "SCENECODE_QWEN_REASONING", "SCENECODE_THINKING_BUDGET",
                "SCENECODE_MAX_TOKENS", "SCENECODE_ASSET_VALIDATION", "SCENECODE_PILOT_ONLY")},
            "forced_tool_reasoning": "none", "job_id": os.getenv("SLURM_JOB_ID"),
            "note": "Source snapshot, not a separate installed environment. No authentication variables recorded."}
        write_json(destination / "manifest.json", manifest)
    status = json.loads(status_path.read_text()) if status_path.exists() else {"started_at": now}
    status.setdefault("started_at", now)
    status.update(updated_at=now, stage=args.stage, job_id=os.getenv("SLURM_JOB_ID"),
                  model=os.getenv("SCENECODE_QWEN_MODEL"), output=str(args.root.resolve()))
    if args.exit_code is None:
        status["state"] = "running"
    else:
        status.update(exit_code=args.exit_code, finished_at=now,
                      state="completed" if args.exit_code == 0 else "failed",
                      elapsed_seconds=(datetime.fromisoformat(now) - datetime.fromisoformat(status["started_at"])).total_seconds())
        manifest_path = args.root / "source_snapshot/manifest.json"
        if manifest_path.exists():
            hashes = json.loads(manifest_path.read_text())["sha256"]
            status["source_changed_since_start"] = [name for name, digest in hashes.items()
                if not (repo / name).is_file() or hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest]
        summary_path = args.root / "run_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            status.update(room_validated=summary["room_validated"],
                          verified_correct_result=summary["verified_correct_result"],
                          visual_review_required=not bool((summary.get("visual_review") or {}).get("passed")))
    write_json(status_path, status)
    print(f"Run status: {args.stage} -> {status['state']} ({status_path})", flush=True)


if __name__ == "__main__":
    main()
