"""Draft bounded generated-TRAIN FK supervisor, not deployed or accepted.

The launch bridge, witness acceptance, full source binding and process-level
fixtures remain to be implemented before this module can be used.
"""
import sys
if sys.flags.optimize:
    raise RuntimeError('optimized Python disables required checks')
if not sys.flags.isolated:
    raise RuntimeError('isolated supervisor Python is required')

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import hashlib
import pathlib
import time

# Authenticate the one-shot request and every declared source before importing
# project modules. The expected request digest is supplied by the launcher.
pre = argparse.ArgumentParser(add_help=False)
pre.add_argument('--request', required=True)
pre.add_argument('--trusted-request-sha256', required=True)
pre_args, _ = pre.parse_known_args()
base = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
req = pathlib.Path(pre_args.request)
if req != base / 'receipts/user_augmentation2526_fk_author_v2_request.json' or req.resolve() != req or req.lstat().st_uid != 1013:
    raise RuntimeError('invalid author request path')
raw = req.read_bytes()
if hashlib.sha256(raw).hexdigest() != pre_args.trusted_request_sha256:
    raise RuntimeError('request changed after launch')
bootstrap = json.loads(raw)
for name, digest in bootstrap['bound_sources'].items():
    path = base / name
    if path.parent != base / 'scripts' or path.resolve() != path or path.lstat().st_uid != 1013:
        raise RuntimeError('unsafe dependency path')
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError('dependency changed after launch')
cache = base / 'receipts/augmentation2526_no_bytecode_cache_v1'
if cache.exists() or cache.is_symlink():
    raise RuntimeError('isolated bytecode cache path must not exist')
sys.pycache_prefix = str(cache)
sys.dont_write_bytecode = True
sys.path.append(str(base / 'scripts'))

from step4_train_only_protocol import ROOT, RESERVE, LOG, owned, read, sha, atomic
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised

ASSET = 128 * 1024**2
STATES = 6 * 128 * 1024**2
RAM = 16 * 1024**3
GPU = 20 * 1024**3


def remaining_upper(selected, completed):
    """Keep the full bound for every clip not yet durably journaled."""
    assert type(completed) is int and 0 <= completed <= len(selected)
    return sum(item['frames'] * 2152 + 8192 for item in selected[completed:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--trusted-request-sha256', required=True)
    sp = owned(parser.parse_args().request)
    s = read(sp)
    assert os.getuid() == 1013 and sha(__file__) == s['supervisor_sha256']
    assert s['fk_authorized'] is True and s['role'] == 'author'
    allowed = {'author': 'user_augmentation2526_fk_author_v2',
               'document': 'user_augmentation2526_fk_document_v2'}
    assert s['run'] == allowed[s['role']] and s['output_root'] == str(ROOT / 'step34_20261029')
    assert s['future_state_upper_bytes'] == STATES
    for name, digest in s['bound_sources'].items():
        assert sha(owned(ROOT / name)) == digest
    assert sha(owned(ROOT / 'scripts/export_augmentation2526_train_fk_v2.py')) == s['executor_sha256']
    lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
    deadline = time.monotonic() + 600
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError('shared model lock unavailable for 600 seconds')
            time.sleep(1)
    from augmentation2526_integrity_fk_inputs_v1 import accepted_generated_fk_inputs
    metadata = accepted_generated_fk_inputs(s['audit_binding'])
    assert metadata['binding'] == s['input_binding']
    all_clips = metadata['plan']['inputs']
    assert len(all_clips) == 2526 and metadata['plan']['output_npz_upper_bytes'] == 5764264784
    clips = all_clips if s['role'] == 'library' else all_clips[:1]
    assert s['output_upper_bytes'] == remaining_upper(clips, 0) + ASSET
    # Library must later bind two accepted real witnesses. A non-null value is
    # not acceptance; do not enable this draft without an actual verifier.
    if s['role'] == 'library':
        assert s['witness_binding'] is not None
    parent = owned(s['output_root'])
    out = parent / s['run']
    log = parent / (s['run'] + '_supervision')
    assert not out.exists() and not log.exists()
    floor = RESERVE + LOG + ASSET + STATES + remaining_upper(clips, 0)
    assert shutil.disk_usage(parent).free >= floor
    pre = sample_resources(ROOT, None, RAM, floor)
    gpu = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2
    assert gpu >= GPU
    log.mkdir(mode=0o700, exist_ok=False)

    def sample(pid):
        completed = 0
        rp = out / 'receipt.json'
        if rp.exists():
            rec = read(rp)
            completed = rec['exported_clips']
            assert type(completed) is int and 0 <= completed <= len(clips)
            journal = out / 'references.private.jsonl'
            if completed:
                rows = [json.loads(line) for line in owned(journal).read_text().splitlines()]
                assert len(rows) >= completed
                seen = set()
                for i in range(completed):
                    row, clip = rows[i], clips[i]
                    path = owned(row['file'])
                    assert row['index'] == clip['index'] and row['frames'] == clip['frames']
                    assert path.parent == out and path not in seen
                    assert path.stat().st_size == row['bytes'] <= clip['frames'] * 2152 + 8192
                    seen.add(path)
        minimum = RESERVE + LOG + ASSET + STATES + remaining_upper(clips, completed)
        value = sample_resources(ROOT, pid, RAM, minimum)
        direct_log = log / 'child.log'
        value['direct_log_bytes'] = direct_log.stat().st_size if direct_log.exists() else 0
        assert value['direct_log_bytes'] <= LOG
        value['completed_clips_accounted'] = completed
        value['remaining_npz_upper_bytes'] = remaining_upper(clips, completed)
        return value

    command = ['taskset', '-c', '0-3', '/usr/bin/env', '-u', 'PYTHONOPTIMIZE', '-u', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER=PCI_BUS_ID', 'PYTHONDONTWRITEBYTECODE=1',
               'PYTHONIOENCODING=utf-8', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
               'bash', str(ROOT / 'scripts/run_tracker.sh'),
               str(ROOT / 'scripts/export_augmentation2526_train_fk_v2.py'),
               '--request', str(sp), '--device', 'cuda:1', '--headless']
    proof = run_supervised(command, log / 'child.log', log / 'resources.jsonl',
                           log / 'process_exit.json', sample,
                           72*3600 if s['role'] == 'library' else 900, 15)
    rp = out / 'receipt.json'
    rec = read(rp) if rp.exists() else {}
    proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
                 receipt_sha256=sha(rp) if rp.exists() else None,
                 resource_sha256=sha(log / 'resources.jsonl'),
                 process_exit_sha256=sha(log / 'process_exit.json'),
                 preflight_ram_bytes=pre['available_host_ram_bytes'], preflight_gpu_free_bytes=gpu,
                 status=rec.get('status'))
    atomic(log / 'exit.json', proof)
    assert proof['supervision_passed'] is True and proof['child_exit_code'] == 0
    assert rec['status'] == 'AUGMENTATION_TRAIN_ISAAC_REFERENCES_FINISHED'
    assert rec['output_writers_closed'] is True and rec['exported_clips'] == len(clips)


if __name__ == '__main__':
    main()
