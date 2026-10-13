"""Cancel only unfinished render jobs recorded under this parent run."""
import json
import subprocess
import sys
from pathlib import Path

for path in Path(sys.argv[1]).glob('*/dispatch.json'):
    record=json.loads(path.read_text())
    if record['state']=='submitted' and str(record['job_id']).isdigit():
        subprocess.run(['scancel',str(record['job_id'])],capture_output=True,text=True)
        record['state']='cancelled_on_parent_exit'
        path.write_text(json.dumps(record,indent=2)+'\n')
