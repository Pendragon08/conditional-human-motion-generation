"""Serial headless generated FK, bounded to7200s with persistent HOME outputs."""
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
sp = owned(a.request)
s = read(sp)
assert sha(__file__) == s['supervisor_sha256']
for name, digest in s['bound_sources'].items():
    assert sha(owned(ROOT / name)) == digest, name
assert sha(ROOT / 'scripts/export_ldm100k_generated_fk.py') == s['executor_sha256']
assert sha(ROOT / 'scripts/run_tracker.sh') == s['wrapper_sha256']
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
summary_receipt = owned(s['summary_receipt'], pathlib.Path('/home/ouyanghao'))
assert sha(summary_receipt) == s['summary_receipt_sha256']
assert read(summary_receipt)['output_writers_closed']
for suffix in ['_process_exit.json', '_exit.json', '_outer_exit.json']:
    path = owned(ROOT / 'receipts' / ('ldm100k_diagnostic_summary_' + s['summary_run'] + suffix))
    assert sha(path) == s['summary_closure_sha256'][suffix]
    value = read(path)
    assert value.get('actual_returncode', 0) == 0 and value.get('child_exit_code', 0) == 0
    assert value.get('supervision_passed', True) and value.get('child_reaped', True)
parent = owned(s['output_root'], pathlib.Path('/home/ouyanghao'))
assert not (parent / s['run']).exists()
assert shutil.disk_usage(parent).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + s['output_upper_bytes']
logs = parent / (s['run'] + '_supervision')
logs.mkdir(mode=0o700, exist_ok=False)


def sample(pid):
    value = sample_resources(ROOT, pid, 16 * 1024**3, 2 * 1024**3 + 128 * 1024**2)
    value['home_free_bytes'] = shutil.disk_usage(parent).free
    assert value['home_free_bytes'] >= s['reserve_bytes'] + s['log_allowance_bytes']
    return value


cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'PYTHONIOENCODING=utf-8', 'OMP_NUM_THREADS=4',
       'OPENBLAS_NUM_THREADS=4', 'bash', str(ROOT / 'scripts/run_tracker.sh'),
       str(ROOT / 'scripts/export_ldm100k_generated_fk.py'), '--request', str(sp), '--device', 'cuda:1', '--headless']
proof = run_supervised(cmd, logs / 'child.log', logs / 'resources.jsonl', logs / 'process_exit.json', sample, 7200, 15)
rp = parent / s['run'] / 'receipt.json'
rec = read(rp) if rp.exists() else {}
proof.update(run=s['run'], status=rec.get('status'), source_sha256=sha(__file__), request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(logs / 'resources.jsonl'),
    process_exit_sha256=sha(logs / 'process_exit.json'), supervision_dir=str(logs), new_neural_model_loads=0)
with (logs / 'exit.json').open('x') as f:
    json.dump(proof, f, indent=2, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
print(json.dumps(proof), flush=True)
assert proof['supervision_passed'] and rec['status'] == 'USER_LDM100K_GENERATED_ISAAC_FK_FINISHED'
assert rec['output_writers_closed'] and rec['exported_trials'] + rec['no_transition_trials'] == 1880
