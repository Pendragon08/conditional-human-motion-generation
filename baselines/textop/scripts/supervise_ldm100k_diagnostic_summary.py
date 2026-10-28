"""Bound the independent CPU summary; never discovers or stops another job."""
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
assert sha(owned(ROOT / 'scripts/summarize_ldm100k_diagnostic.py')) == s['executor_sha256']
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
prefix = 'ldm100k_diagnostic_summary_' + s['run']
cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONIOENCODING=utf-8',
       'OPENBLAS_NUM_THREADS=4', 'OMP_NUM_THREADS=4', str(ROOT / 'env/bin/python'), '-B',
       str(ROOT / 'scripts/summarize_ldm100k_diagnostic.py'), '--request', str(request)]
def resources(pid):
    value = sample_resources(ROOT, pid, 16 * 1024**3, s['disk_floor_bytes'])
    value['output_fs_free_bytes'] = shutil.disk_usage(s['output_root']).free
    assert value['output_fs_free_bytes'] >= 2 * 1024**3 + s['metadata_upper_bytes'] + 128 * 1024**2
    return value


proof = run_supervised(cmd, ROOT / 'logs' / (prefix + '.log'), ROOT / 'logs' / (prefix + '_resources.jsonl'),
    ROOT / 'receipts' / (prefix + '_process_exit.json'),
    resources, 900, 15)
rp = pathlib.Path(s['output_root']) / s['run'] / 'receipt.json'
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(request),
    receipt_sha256=sha(rp) if rp.exists() else None,
    resource_sha256=sha(ROOT / 'logs' / (prefix + '_resources.jsonl')))
with (ROOT / 'receipts' / (prefix + '_exit.json')).open('x') as f:
    json.dump(proof, f, indent=2, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
assert proof['supervision_passed']
assert read(rp)['status'] == 'USER_LDM100K_DIAGNOSTIC32_CPU_SUMMARY_FINISHED'
