"""Bound pinned public text-model acquisition and frozen CPU checks; zero evaluator gradients."""
import argparse
import fcntl
import json
import os
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
assert sha(ROOT / 'scripts/prepare_tmr_text_models.py') == s['executor_sha256']
assert sha(owned(s['runtime_receipt']['path'])) == s['runtime_receipt']['sha256']
assert s['pretrained_weight_downloads_authorized'] and len(s['models']) == 2
assert not s['evaluator_training_authorized']
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
parent = owned(ROOT / 'evaluator_tmr_20261029')
assert not (parent / s['run']).exists()
assert shutil.disk_usage(parent).free > 2 * 1024**3 + 128 * 1024**2 + s['output_upper_bytes']
log = parent / (s['run'] + '_supervision')
log.mkdir(mode=0o700, exist_ok=False)


def sample(pid):
    return sample_resources(ROOT, pid, 16 * 1024**3, 2 * 1024**3 + 128 * 1024**2)


cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONIOENCODING=utf-8',
       'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
       str(ROOT / 'env/bin/python'), '-B', str(ROOT / 'scripts/prepare_tmr_text_models.py'),
       '--request', str(sp)]
proof = run_supervised(cmd, log / 'child.log', log / 'resources.jsonl', log / 'process_exit.json', sample, 3600, 15)
rp = parent / s['run'] / 'receipt.json'
r = read(rp) if rp.exists() else {}
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(log / 'resources.jsonl'),
    process_exit_sha256=sha(log / 'process_exit.json'))
with (log / 'exit.json').open('x') as f:
    json.dump(proof, f, indent=2, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
assert proof['supervision_passed'] and r['status'] == 'PINNED_TMR_TEXT_MODELS_CPU_VERIFIED'
assert r['output_writers_closed'] and not r['capability_probe']['cuda_initialized']
assert r['training_optimizer_updates'] == 0 and not r['formal_tmr_metrics_available']
