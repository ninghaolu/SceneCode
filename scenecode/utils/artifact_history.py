"""Opt-in immutable intermediate artifacts for model experiments."""
import json
import os
import time
import uuid
from pathlib import Path


def snapshot(path, content, *, suffix=None):
    if not os.getenv("SCENECODE_AUDIT_DIR"):
        return None
    path = Path(path)
    history = path.parent / "attempt_history"
    history.mkdir(parents=True, exist_ok=True)
    destination = history / f"{path.stem}_{time.time_ns()}_{uuid.uuid4().hex}{suffix or path.suffix}"
    destination.write_text(content if isinstance(content, str) else json.dumps(content, indent=2, default=str))
    return destination
