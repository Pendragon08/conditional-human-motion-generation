"""Detach the documented --run invocation once; no automatic restart.

The experiment protocol/source stay frozen. Persist the launcher PID and exact
command so an interrupted SSH observation cannot accidentally start a new job.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from verify_execution_lineage_v39_20261012 import read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/paired_feedback_validation_v51_20261012'
witness = read(OUT / 'DOCUMENT_WITNESS.json')
assert witness['status'] == 'PASS'
assert witness['runtime_witness_sha256'] == sha(OUT / 'runtime_witness.json')
assert read(OUT / 'state.json')['status'] == 'WITNESS_COMPLETE_PENDING_INDEPENDENT_REVIEW'
receipt = OUT / 'launcher_receipt.json'
assert not receipt.exists(), 'Existing launch receipt: inspect its process, never restart blindly'
command = [sys.executable, str(ROOT / 'scripts/run_paired_feedback_validation_v51_20261012.py'), '--run']
with (OUT / 'launcher.log').open('x') as log:
    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
result = dict(pid=process.pid, command=command, launched_at=time.time(),
    source_sha256=sha(Path(__file__)), versions_sha256=sha(OUT / 'versions.json'),
    witness_sha256=sha(OUT / 'DOCUMENT_WITNESS.json'),
    role='Detached exact documented main argv; launch alone does not prove live or complete')
with receipt.open('x') as file:
    json.dump(result, file, indent=2)
print(json.dumps(result), flush=True)
