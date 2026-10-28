"""Bounded stdlib metadata builder; no model/GPU/SDK work is dispatched."""
import argparse
import fcntl
import json
import os
import shutil
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from ldm100k_verified_input import ROOT, read, sha, owned, accepted_verified_input
from ldm100k_scope_contract import RUN, validate_policy, validate_scope_receipt

p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
a = p.parse_args()
assert os.getuid() == 1013 and a.run == RUN
sp = owned(ROOT / 'configs/ldm100k-heldout-scope-spec.json')
s = read(sp)
validate_policy(s)
assert sha(__file__) == s['supervisor_sha256']
assert sha(ROOT / 'scripts/build_ldm100k_heldout_scope.py') == s['executor_sha256']
for name, digest in s['bound_sources'].items():
    assert sha(ROOT / name) == digest, name
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
accepted_verified_input(s['verified_model'])
assert shutil.disk_usage(ROOT).free >= s['minimum_disk_free_bytes']
parent = owned(ROOT / 'ldm100k_user_handoff_20261028')
out = parent / a.run
assert not out.exists()
prefix = 'ldm100k_heldout_scope_' + a.run
ep = ROOT / 'receipts' / (prefix + '_exit.json')
assert not ep.exists()
cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=',
       'PYTHONIOENCODING=utf-8', 'PYTHONUNBUFFERED=1', 'OPENBLAS_NUM_THREADS=4',
       'OMP_NUM_THREADS=4', 'python3', str(ROOT / 'scripts/build_ldm100k_heldout_scope.py'),
       '--run', a.run]
resources = ROOT / 'logs' / (prefix + '_resources.jsonl')
process = ROOT / 'receipts' / (prefix + '_process_exit.json')
proof = run_supervised(cmd, ROOT / 'logs' / (prefix + '.log'), resources, process,
                       lambda pid: sample_resources(ROOT, pid, s['minimum_ram_free_bytes'],
                                                    s['minimum_disk_free_bytes']),
                       s['timeout_seconds'], 15)
rp = out / 'receipt.json'
rec = read(rp) if rp.exists() else {}
proof.update(run=a.run, status=rec.get('status'), receipt_sha256=sha(rp) if rp.exists() else None,
             spec_sha256=sha(sp), source_sha256=sha(__file__), resource_sha256=sha(resources),
             process_exit_sha256=sha(process), new_gpu_contexts=0, new_sdk_tasks=0,
             training_optimizer_updates_performed=0)
with ep.open('x') as handle:
    json.dump(proof, handle, indent=2, allow_nan=False)
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps(proof), flush=True)
assert proof['supervision_passed']
validate_scope_receipt(rec, s, sha(sp))
