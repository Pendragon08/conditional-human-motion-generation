"""Draft bounded shared-lock supervisor for the generated FK array audit."""
import sys
if sys.flags.optimize or not sys.flags.isolated:
    raise RuntimeError('non-optimized isolated Python required')

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import time

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN = 'user_augmentation2526_fk_terminal_audit_v1'
ENTRY = ROOT / 'scripts/run_augmentation2526_fk_terminal_audit_v1.py'
parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
parser.add_argument('--trusted-request-sha256', required=True)
a = parser.parse_args()
request = pathlib.Path(a.request)
if request != ROOT / 'receipts/user_augmentation2526_fk_terminal_audit_v1_request.json' or request.resolve() != request or request.lstat().st_uid != 1013:
    raise ValueError('invalid audit request')
raw = request.read_bytes()
if hashlib.sha256(raw).hexdigest() != a.trusted_request_sha256:
    raise ValueError('audit request changed')
s = json.loads(raw)
if s['schema'] != 'textop.augmentation2526_fk_terminal_audit.v1' or s['authorized'] is not True or s['run'] != RUN:
    raise ValueError('audit request unauthorized')
for name, digest in s['bound_sources'].items():
    path = ROOT / name
    if path.parent != ROOT / 'scripts' or path.resolve() != path or path.lstat().st_uid != 1013 or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('audit source changed')
cache = ROOT / 'receipts/augmentation2526_no_bytecode_cache_v1'
if cache.exists() or cache.is_symlink():
    raise ValueError('audit cache isolation unavailable')
sys.pycache_prefix = str(cache)
sys.dont_write_bytecode = True
sys.path.append(str(ROOT / 'scripts'))
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised
from filtered15000_vae_completion import owned, read, sha

lock = owned(ROOT / 'receipts/full_library_capacity.lock').open('r+')
deadline = time.monotonic() + 600
while True:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except BlockingIOError:
        if time.monotonic() >= deadline:
            raise TimeoutError('shared model lock unavailable')
        time.sleep(1)
parent = owned(ROOT / 'step34_20261029')
out = parent / RUN
log = parent / (RUN + '_supervision')
if out.exists() or log.exists():
    raise FileExistsError('existing audit output must be retained')
floor = 2 * 1024**3 + 128 * 1024**2 + 32 * 1024**2 + 128 * 1024**2 + 6 * 128 * 1024**2
if shutil.disk_usage(ROOT).free < floor:
    raise RuntimeError('insufficient data capacity')
sample_resources(ROOT, None, 16 * 1024**3, floor)
log.mkdir(mode=0o700)


def sample(pid):
    value = sample_resources(ROOT, pid, 16 * 1024**3, floor)
    path = log / 'child.log'
    value['direct_log_bytes'] = path.stat().st_size if path.exists() else 0
    if value['direct_log_bytes'] > 128 * 1024**2:
        raise RuntimeError('audit log budget exceeded')
    return value


command = ['taskset', '-c', '0-3', '/usr/bin/env', '-u', 'PYTHONOPTIMIZE',
           'CUDA_VISIBLE_DEVICES=', '/usr/bin/python3', '-I', '-B', str(ENTRY),
           '--request', str(request), '--trusted-request-sha256', a.trusted_request_sha256]
proof = run_supervised(command, log / 'child.log', log / 'resources.jsonl',
                       log / 'process_exit.json', sample, 7200, 15)
receipt_path = out / 'receipt.json'
receipt = read(receipt_path) if receipt_path.exists() else {}
proof.update(run=RUN, request_sha256=sha(request), source_sha256=sha(__file__),
             receipt_sha256=sha(receipt_path) if receipt_path.exists() else None,
             resource_sha256=sha(log / 'resources.jsonl'),
             process_exit_sha256=sha(log / 'process_exit.json'), status=receipt.get('status'))
with (log / 'exit.json').open('x', encoding='utf-8') as f:
    json.dump(proof, f, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
if proof['supervision_passed'] is not True or proof['child_exit_code'] != 0 or receipt.get('status') != 'GENERATED_TRAIN_FK_TERMINAL_AUDIT_FINISHED':
    raise RuntimeError('terminal audit did not close successfully')
