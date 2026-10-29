"""Serialize and resource-bound only the new resume/witness process group."""
import argparse
import fcntl
import json
import os
import pathlib
import shutil
import subprocess
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from tmr100_capacity_resume_protocol import ROOT, PARENT, HOME, RUNS, owned, read, sha, origin

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
p.add_argument('--role', choices=list(RUNS), required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert sha(__file__) == s['supervisor_sha256'] and os.getuid() == 1013
assert sha(ROOT / 'scripts/run_tmr100_capacity_resume.py') == s['executor_sha256']
assert s['formal_training_authorized'] if a.role == 'train' else s['witnesses_authorized']
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
origin(s)
run = RUNS[a.role]
assert not (PARENT / run).exists()
log = PARENT / (run + '_supervision')
log.mkdir(mode=0o700, exist_ok=False)
home = owned(s['checkpoint_dir'], HOME)
assert subprocess.check_output(['findmnt', '-T', str(home), '-n', '-o', 'FSTYPE'], text=True).strip() == 'ext4'
if a.role == 'train':
    assert not list(home.iterdir())
    assert shutil.disk_usage(home).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + 2 * s['checkpoint_upper_bytes']
gpu = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2
assert gpu >= 20 * 1024**3


def sample(pid):
    value = sample_resources(ROOT, pid, 16 * 1024**3, 2 * 1024**3 + 128 * 1024**2)
    value['home_free_bytes'] = shutil.disk_usage(home).free
    assert value['home_free_bytes'] >= s['reserve_bytes'] + s['log_allowance_bytes']
    return value


cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=1', 'PYTHONIOENCODING=utf-8',
    'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8',
    'TOKENIZERS_PARALLELISM=false', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1',
    str(ROOT / 'env/bin/python'), '-B', str(ROOT / 'scripts/run_tmr100_capacity_resume.py'), '--request', str(sp), '--role', a.role]
proof = run_supervised(cmd, log / 'child.log', log / 'resources.jsonl', log / 'process_exit.json', sample,
    259200 if a.role == 'train' else 900, 15, grace_seconds=30)
rp = PARENT / run / 'receipt.json'
rec = read(rp) if rp.exists() else {}
proof.update(run=run, request_path=str(sp), request_sha256=sha(sp), source_sha256=sha(__file__),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(log / 'resources.jsonl'),
    process_exit_sha256=sha(log / 'process_exit.json'))
with (log / 'exit.json').open('x') as f:
    json.dump(proof, f, indent=2); f.flush(); os.fsync(f.fileno())
expected = 'TMR100_CAPACITY_RESUMED_100_EPOCHS_FINISHED' if a.role == 'train' else 'TMR100_CAPACITY_RESUME_REAL_GPU_WITNESS_VERIFIED'
assert proof['supervision_passed'] and rec['status'] == expected and rec['output_writers_closed']
