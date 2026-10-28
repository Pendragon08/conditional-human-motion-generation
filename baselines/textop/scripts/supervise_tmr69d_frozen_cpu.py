"""Bound one future CPU evaluator export or fresh check after training closes."""
import argparse
import fcntl
import json
import os
import pathlib
import shutil
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from tmr69d_trained_acceptance import ROOT, HOME, PARENT, owned, read, sha, accepted_training, accepted_frozen_export

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['supervisor_sha256'] and s['authorized'] is True
assert sha(owned(ROOT/'scripts/run_tmr69d_frozen_cpu.py')) == s['executor_sha256']
for name, value in s['bound_sources'].items():
    assert sha(owned(ROOT/name)) == value
lock = (ROOT/'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
accepted_training(s['training_binding'])
if s['role'] == 'verify':
    accepted_frozen_export(s['export_binding'])
parent = owned(PARENT)
assert not (parent/s['run']).exists()
log = parent/(s['run']+'_supervision')
log.mkdir(mode=0o700, exist_ok=False)
home = owned(pathlib.Path(s['inference_path']).parent, HOME)


def sample(pid):
    result = sample_resources(ROOT, pid, 16*1024**3, 2*1024**3+128*1024**2)
    result['home_free_bytes'] = shutil.disk_usage(home).free
    assert result['home_free_bytes'] >= s['reserve_bytes']+s['home_log_allowance_bytes']
    return result


command = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONIOENCODING=utf-8',
    'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
    str(ROOT/'env/bin/python'), '-B', str(ROOT/'scripts/run_tmr69d_frozen_cpu.py'), '--request', str(sp)]
proof = run_supervised(command, log/'child.log', log/'resources.jsonl', log/'process_exit.json', sample, 900, 15)
rp = parent/s['run']/'receipt.json'
rec = read(rp) if rp.exists() else {}
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(log/'resources.jsonl'),
    process_exit_sha256=sha(log/'process_exit.json'))
with (log/'exit.json').open('x') as stream:
    json.dump(proof, stream, indent=2, allow_nan=False)
    stream.flush()
    os.fsync(stream.fileno())
expected = 'TMR69D_TRAINED_FROZEN_CPU_EXPORTED' if s['role'] == 'export' else 'TMR69D_TRAINED_FROZEN_FRESH_CPU_VERIFIED'
assert proof['supervision_passed'] and rec['status'] == expected and rec['output_writers_closed']
