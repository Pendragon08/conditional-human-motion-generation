"""One bounded strict replay group; no training and no overlapping model/SDK."""
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
assert os.getuid() == 1013
request = owned(a.request)
s = read(request)
assert sha(__file__) == s['supervisor_sha256']
for name, digest in s['bound_sources'].items():
    assert sha(owned(ROOT / name)) == digest, name
assert sha(ROOT / 'scripts/evaluate_ldm100k_generated_physics.py') == s['executor_sha256']
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
parent = owned(s['output_root'], pathlib.Path('/home/ouyanghao'))
rp = owned(s['fk_receipt'], pathlib.Path('/home/ouyanghao'))
assert sha(rp) == s['fk_receipt_sha256'] and read(rp)['output_writers_closed']
for path, digest in s['fk_closure'].items():
    assert sha(owned(path, pathlib.Path('/home/ouyanghao') if path.startswith('/home/') else ROOT)) == digest
assert not (parent / s['run']).exists()
assert shutil.disk_usage(parent).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + s['output_upper_bytes']
logs = parent / (s['run'] + '_supervision')
logs.mkdir(mode=0o700, exist_ok=False)


def sample(pid):
    v = sample_resources(ROOT, pid, 16 * 1024**3, 2 * 1024**3 + 128 * 1024**2)
    v['home_free_bytes'] = shutil.disk_usage(parent).free
    assert v['home_free_bytes'] >= s['reserve_bytes'] + s['log_allowance_bytes']
    return v


cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'PYTHONIOENCODING=utf-8', 'OMP_NUM_THREADS=4',
       'OPENBLAS_NUM_THREADS=4', 'bash', str(ROOT / 'scripts/run_tracker.sh'),
       str(ROOT / 'scripts/evaluate_ldm100k_generated_physics.py'), '--request', str(request), '--device', 'cuda:1', '--headless']
proof = run_supervised(cmd, logs / 'child.log', logs / 'resources.jsonl', logs / 'process_exit.json', sample, 7200, 15)
rp = parent / s['run'] / 'receipt.json'
rec = read(rp) if rp.exists() else {}
proof.update(run=s['run'], status=rec.get('status'), source_sha256=sha(__file__), request_sha256=sha(request),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(logs / 'resources.jsonl'),
    process_exit_sha256=sha(logs / 'process_exit.json'), supervision_dir=str(logs))
with (logs / 'exit.json').open('x') as f:
    json.dump(proof, f, indent=2, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
print(json.dumps(proof), flush=True)
assert proof['supervision_passed'] and rec['status'] == 'USER_LDM100K_GENERATED_STRICT_PHYSICS_FINISHED'
assert rec['output_writers_closed'] and rec['finished_trials'] == 1880 and rec['optimizer_updates'] == 0
