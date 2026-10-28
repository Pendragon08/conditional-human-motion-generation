"""Bound actual TRAIN NPZ and normalized69D acceptance; zero learned encoders or gradients."""
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
assert sha(ROOT / 'scripts/accept_tmr69d_numeric_data.py') == s['executor_sha256']
assert sha(ROOT / 'scripts/paper69_runtime.py') == s['codec_sha256']
assert not s['training_authorized']
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
parent = owned(ROOT / 'evaluator_tmr_20261029')
assert not (parent / s['run']).exists()


def pending_public_bytes():
    incoming = parent / 'text_models_import_v1'
    intent = incoming / 'import_intent.json'
    if not intent.exists():
        return 0
    total = 0
    for item in read(intent)['files']:
        p = incoming / item['relative']
        if not p.exists():
            p = incoming / (item['relative'] + '.partial')
        count = p.stat().st_size if p.exists() else 0
        assert 0 <= count <= item['bytes']
        total += item['bytes'] - count
    return total

assert shutil.disk_usage(parent).free > 2 * 1024**3 + 128 * 1024**2 + s['output_upper_bytes'] + pending_public_bytes()
log = parent / (s['run'] + '_supervision')
log.mkdir(mode=0o700, exist_ok=False)


def sample(pid):
    return sample_resources(ROOT, pid, 16 * 1024**3, 2 * 1024**3 + 128 * 1024**2 + pending_public_bytes())


cmd = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONIOENCODING=utf-8',
       'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
       str(ROOT / 'env/bin/python'), '-B', str(ROOT / 'scripts/accept_tmr69d_numeric_data.py'),
       '--request', str(sp)]
proof = run_supervised(cmd, log / 'child.log', log / 'resources.jsonl', log / 'process_exit.json', sample, 900, 15)
rp = parent / s['run'] / 'receipt.json'
r = read(rp) if rp.exists() else {}
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(log / 'resources.jsonl'),
    process_exit_sha256=sha(log / 'process_exit.json'))
with (log / 'exit.json').open('x') as f:
    json.dump(proof, f, indent=2, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
assert proof['supervision_passed'] and r['status'] == 'TMR69D_NUMERIC_TRAIN_INPUTS_ACCEPTED'
assert r['output_writers_closed'] and r['original_train_sources'] == 5361 and r['encoder_weight_loads'] == r['optimizer_updates'] == 0 and not r['cuda_initialized']
