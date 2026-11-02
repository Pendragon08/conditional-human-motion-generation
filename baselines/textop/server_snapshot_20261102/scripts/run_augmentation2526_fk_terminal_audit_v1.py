"""Draft isolated entry for the complete generated FK library audit."""
import sys
if sys.flags.optimize or not sys.flags.isolated:
    raise RuntimeError('non-optimized isolated Python required')

import argparse
import hashlib
import json
import os
import pathlib

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN = 'user_augmentation2526_fk_terminal_audit_v1'
SUPERVISOR = ROOT / 'scripts/supervise_augmentation2526_fk_terminal_audit_v1.py'
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
from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor

pid = os.getppid()
found = False
for _ in range(8):
    if pid <= 1:
        break
    proc = pathlib.Path('/proc') / str(pid)
    argv = (proc / 'cmdline').read_bytes().decode().split('\0')
    if proc.stat().st_uid == 1013 and str(request) in argv and str(SUPERVISOR) in argv:
        found = True
        break
    pid = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
if not found:
    raise RuntimeError('dedicated audit supervisor required')
lock = ROOT / 'receipts/full_library_capacity.lock'
st = lock.stat()
if not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(), pid, os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
    raise RuntimeError('audit supervisor does not hold common lock')
if set(os.sched_getaffinity(0)) != {0, 1, 2, 3} or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('audit resource isolation mismatch')
from augmentation2526_fk_terminal_audit_v1 import audit_library

out = ROOT / 'step34_20261029' / RUN
out.mkdir(mode=0o700, exist_ok=False)
report = audit_library()
raw = (json.dumps(report, separators=(',', ':'), allow_nan=False) + '\n').encode()
if len(raw) > s['output_limit_bytes']:
    raise ValueError('audit report exceeds limit')
temporary = out / 'audit.private.json.partial'
with temporary.open('xb') as f:
    os.chmod(temporary, 0o600)
    f.write(raw)
    f.flush()
    os.fsync(f.fileno())
final = out / 'audit.private.json'
temporary.rename(final)
fd = os.open(str(out), os.O_RDONLY)
try:
    os.fsync(fd)
finally:
    os.close(fd)
receipt = dict(status='GENERATED_TRAIN_FK_TERMINAL_AUDIT_FINISHED', run=RUN,
               request_sha256=hashlib.sha256(request.read_bytes()).hexdigest(),
               report_sha256=hashlib.sha256(raw).hexdigest(), report_bytes=len(raw),
               clips=report['clips'], frames=report['frames'], output_writers_closed=True,
               model_loads=0, sdk_loads=0, physical_tracking_accepted=False)
with (out / 'receipt.json').open('x', encoding='utf-8') as f:
    json.dump(receipt, f, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
