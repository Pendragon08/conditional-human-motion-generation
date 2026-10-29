"""New serial PPO job, actual resources/closure and bounded private states."""
import argparse
import fcntl
import os
import shutil
from step4_train_only_protocol import ROOT, RESERVE, LOG, owned, read, sha, atomic
from step4_train_only_ppo_inputs import accepted_library
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised

p = argparse.ArgumentParser(); p.add_argument('--request', required=True); a = p.parse_args()
sp = owned(a.request); s = read(sp)
assert sha(__file__) == s['supervisor_sha256'] and os.getuid() == 1013
for n, h in s['bound_sources'].items(): assert sha(owned(ROOT/n)) == h
lock = (ROOT/'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
accepted_library(s['library_binding'])
parent = owned(s['output_root']); assert not (parent/s['run']).exists()
assert shutil.disk_usage(parent).free >= RESERVE+LOG+s['output_upper_bytes']
log = parent/(s['run']+'_supervision'); log.mkdir(mode=0o700, exist_ok=False)


def sample(pid):
    value = sample_resources(ROOT, pid, 16*1024**3, RESERVE+LOG)
    paths = [log/'child.log', parent/s['run']/'metrics.private.jsonl']
    value['direct_log_and_metrics_bytes'] = sum(p.stat().st_size for p in paths if p.exists())
    assert value['direct_log_and_metrics_bytes'] <= LOG
    return value


cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONIOENCODING=utf-8',
    'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
    'bash', str(ROOT/'scripts/run_tracker.sh'), str(ROOT/'scripts/run_step4_train_only_ppo.py'),
    '--request', str(sp), '--headless', '--device', 'cuda:1']
proof = run_supervised(cmd, log/'child.log', log/'resources.jsonl', log/'process_exit.json',
    sample, 3600 if s['role'] != 'train' else 72*3600, 15, grace_seconds=30)
rp = parent/s['run']/'receipt.json'; rec = read(rp) if rp.exists() else {}
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(log/'resources.jsonl'),
    process_exit_sha256=sha(log/'process_exit.json'), status=rec.get('status'))
atomic(log/'exit.json', proof)
assert proof['supervision_passed'] and proof['child_exit_code'] == 0
assert rec['output_writers_closed'] and rec['status'] in ['STAGE4_TRAIN_ONLY_REAL_PPO_RECOVERY_WITNESS_FINISHED', 'STAGE4_ORIGINAL_TRAIN_ONE_STAGE_PPO_FINISHED']
