"""Fail before generation unless this rendering implementation passed real GPU smoke checks."""
import hashlib
import json
import sys
from pathlib import Path

report=json.loads(Path(sys.argv[1]).read_text())
required={'stateful_blender','floor_observation','room_observation','asset_observation','final_export'}
assert report['passed'] and required<=report['checks'].keys()
assert all(report['checks'][name]['passed'] for name in required)
root=Path(__file__).resolve().parents[1]
for name,expected in report['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,f'Render implementation changed since smoke: {name}'
print('CPU/GPU rendering smoke gate passed')
