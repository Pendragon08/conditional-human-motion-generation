"""Bounded serial TRAIN-only FK, with all source and resource receipts."""
import argparse
import fcntl
import os
import shutil
from step4_train_only_protocol import ROOT, RESERVE, LOG, owned, read, sha, atomic
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised

p = argparse.ArgumentParser(); p.add_argument('--request', required=True); a = p.parse_args()
sp = owned(a.request); s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['supervisor_sha256']
for n, h in s['bound_sources'].items(): assert sha(owned(ROOT/n)) == h
lock = (ROOT/'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
parent = owned(s['output_root'])
assert not (parent/s['run']).exists()
assert shutil.disk_usage(parent).free >= RESERVE+LOG+s['output_upper_bytes']
log = parent/(s['run']+'_supervision'); log.mkdir(mode=0o700, exist_ok=False)


def sample(pid):
    value = sample_resources(ROOT, pid, 16*1024**3, RESERVE+LOG)
    value['output_fs_free_bytes'] = shutil.disk_usage(parent).free
    assert value['output_fs_free_bytes'] >= RESERVE+LOG
    return value


command = ['taskset', '-c', '0-3', '/usr/bin/env', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONIOENCODING=utf-8',
    'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'bash', str(ROOT/'scripts/run_tracker.sh'),
    str(ROOT/'scripts/export_step4_train_only_references.py'), '--request', str(sp), '--device', 'cuda:1', '--headless']
proof = run_supervised(command, log/'child.log', log/'resources.jsonl', log/'process_exit.json',
    sample, 7200 if s['role'] == 'library' else 900, 15)
rp = parent/s['run']/'receipt.json'; rec = read(rp) if rp.exists() else {}
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(log/'resources.jsonl'),
    process_exit_sha256=sha(log/'process_exit.json'), status=rec.get('status'))
atomic(log/'exit.json', proof)
assert proof['supervision_passed'] and proof['child_exit_code'] == 0
assert rec['status'] == 'STAGE4_ORIGINAL_TRAIN_ISAAC_REFERENCES_FINISHED' and rec['output_writers_closed']
