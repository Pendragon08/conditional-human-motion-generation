"""Bound post-hoc recorded-state FK; no policy or physics replay."""
import argparse
import fcntl
import json
import os
import pathlib
import shutil
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from ldm100k_verified_input import ROOT, owned, read, sha

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['supervisor_sha256']
assert sha(ROOT / 'scripts/measure_ldm100k_tracking_fidelity.py') == s['executor_sha256']
for name, digest in s['bound_sources'].items():
    assert sha(ROOT / name) == digest, name
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
parent = owned(s['output_root'], pathlib.Path('/home/ouyanghao'))
assert not (parent / s['run']).exists()
assert shutil.disk_usage(parent).free > s['reserve_bytes'] + s['log_allowance_bytes'] + s['output_upper_bytes']
log = parent / (s['run'] + '_supervision')
log.mkdir(mode=0o700, exist_ok=False)


def sample(pid):
    row = sample_resources(ROOT, pid, 16 * 1024**3, 2 * 1024**3 + 128 * 1024**2)
    row['home_free_bytes'] = shutil.disk_usage(parent).free
    assert row['home_free_bytes'] > s['reserve_bytes'] + s['log_allowance_bytes']
    return row


cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'PYTHONIOENCODING=utf-8', 'OMP_NUM_THREADS=4',
       'OPENBLAS_NUM_THREADS=4', 'bash', str(ROOT / 'scripts/run_tracker.sh'),
       str(ROOT / 'scripts/measure_ldm100k_tracking_fidelity.py'), '--request', str(sp), '--device', 'cuda:1', '--headless']
proof = run_supervised(cmd, log / 'child.log', log / 'resources.jsonl', log / 'process_exit.json', sample, 900, 15)
rp = parent / s['run'] / 'receipt.json'
r = read(rp) if rp.exists() else {}
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(log / 'resources.jsonl'),
    process_exit_sha256=sha(log / 'process_exit.json'))
with (log / 'exit.json').open('x') as f:
    json.dump(proof, f, indent=2, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
assert proof['supervision_passed'] and r['status'] == 'USER_LDM100K_RECORDED_TRACKING_FIDELITY_FINISHED'
assert r['output_writers_closed'] and r['processed_trials'] == 1880 and r['new_physics_steps'] == 0
