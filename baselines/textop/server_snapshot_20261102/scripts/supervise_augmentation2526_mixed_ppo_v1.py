"""Draft shared-lock supervisor for mixed-source PPO; not yet deployable."""
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
import subprocess
import time

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUNS = {'author': 'user_augmentation2526_mixed_ppo_author_v1',
        'document': 'user_augmentation2526_mixed_ppo_document_v1',
        'train': 'user_augmentation2526_mixed_ppo_train_v1'}
parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
parser.add_argument('--trusted-request-sha256', required=True)
a = parser.parse_args()
request = pathlib.Path(a.request)
if request.parent != ROOT / 'receipts' or request.resolve() != request or request.lstat().st_uid != 1013:
    raise ValueError('invalid mixed PPO request path')
raw = request.read_bytes()
if hashlib.sha256(raw).hexdigest() != a.trusted_request_sha256:
    raise ValueError('mixed PPO request changed')
s = json.loads(raw)
if s.get('role') not in RUNS or s.get('run') != RUNS[s['role']] or request.name != s['run'] + '_request.json':
    raise ValueError('unexpected mixed PPO role or run')
if s.get('output_root') != str(ROOT / 'step34_20261029'):
    raise ValueError('unexpected mixed PPO output root')
REQUIRED_SOURCES = ['scripts/augmentation2526_fk_plan_v1.py', 'scripts/augmentation2526_integrity_fk_inputs_v1.py', 'scripts/augmentation2526_mixed_inputs_v1.py', 'scripts/augmentation2526_mixed_policy_contract_v1.py', 'scripts/augmentation2526_mixed_ppo_runtime_v1.py', 'scripts/augmentation2526_stop_binding_v1.py', 'scripts/augmentation5368_audit_binding_v1.py', 'scripts/augmentation5368_protocol.py', 'scripts/augmentation5368_verified_inputs_v1.py', 'scripts/augmentation5368_witness_acceptance_v1.py', 'scripts/bounded_headless_close.py', 'scripts/contact_filter_resources.py', 'scripts/contact_process_supervision.py', 'scripts/filter_training_runtime.py', 'scripts/filtered15000_vae_completion.py', 'scripts/ldm100k_control_protocol.py', 'scripts/ldm100k_export_acceptance.py', 'scripts/ldm100k_prefix_completion.py', 'scripts/ldm100k_verified_input.py', 'scripts/quick_pidfd.py', 'scripts/run_augmentation2526_mixed_ppo_v1.py', 'scripts/solid_floor_diagnostic.py', 'scripts/step4_mixed_isaaclab_config_v1.py', 'scripts/step4_train_only_ppo_inputs.py', 'scripts/step4_train_only_protocol.py', 'scripts/supervise_augmentation2526_mixed_ppo_v1.py', 'scripts/vae100k_control_protocol.py', 'scripts/run_tracker.sh']
if not set(REQUIRED_SOURCES).issubset(s['bound_sources']):
    raise ValueError('mixed PPO recursive dependencies are incomplete')
if s['role'] == 'train' and 'scripts/augmentation2526_mixed_ppo_witness_acceptance_v1.py' not in s['bound_sources']:
    raise ValueError('mixed PPO witness verifier missing')
for name, digest in s['bound_sources'].items():
    path = ROOT / name
    if path.parent != ROOT / 'scripts' or path.resolve() != path or path.lstat().st_uid != 1013 or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('mixed PPO source changed')
tracker = 'upstream/TextOpTracker/source/textop_tracker/textop_tracker/'
external = {'configs/tracker-env-spec.json'} | {tracker + name for name in (
    'robots/g1.py', 'tasks/tracking/tracking_env_cfg.py',
    'tasks/tracking/mdp/commands_multi.py', 'tasks/tracking/mdp/observations.py',
    'tasks/tracking/mdp/rewards.py', 'tasks/tracking/mdp/terminations.py',
    'tasks/tracking/config/g1/flat_env_cfg.py',
    'tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py')}
if set(s['external_sources']) != external:
    raise ValueError('mixed PPO tracker source binding incomplete')
for name, digest in s['external_sources'].items():
    path = ROOT / name
    if path.resolve() != path or path.lstat().st_uid != 1013 or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('mixed PPO tracker source changed')
cache = ROOT / 'receipts/augmentation2526_no_bytecode_cache_v1'
if cache.exists() or cache.is_symlink():
    raise ValueError('mixed PPO bytecode cache isolation unavailable')
sys.pycache_prefix = str(cache)
sys.dont_write_bytecode = True
sys.path.append(str(ROOT / 'scripts'))
from augmentation2526_mixed_inputs_v1 import accepted_mixed_library
from augmentation2526_mixed_policy_contract_v1 import validate_policy
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised
from filtered15000_vae_completion import owned, read, sha

policy = validate_policy(s['policy'])
if s['role'] == 'train':
    if policy['training_authorized'] is not True or policy['witnesses_authorized'] is not False:
        raise ValueError('formal mixed PPO is not authorized')
else:
    if policy['training_authorized'] is not False or policy['witnesses_authorized'] is not True:
        raise ValueError('diagnostic mixed PPO is not authorized')
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
accepted, rows = accepted_mixed_library(s['library_binding'])
if accepted['actual_original_train_clips'] != 5551 or accepted['actual_generated_clips'] != 2526 or len(rows) != 8077:
    raise ValueError('mixed library count mismatch')
if s['role'] == 'train':
    # A verified implementation must be supplied before formal training.
    from augmentation2526_mixed_ppo_witness_acceptance_v1 import accepted_mixed_ppo_witnesses
    if accepted_mixed_ppo_witnesses(s['witness_binding']) != s['witness_binding']:
        raise ValueError('mixed PPO witnesses not accepted')
parent = owned(ROOT / 'step34_20261029')
out = parent / s['run']
log = parent / (s['run'] + '_supervision')
if out.exists() or log.exists():
    raise FileExistsError('existing mixed PPO intent must be preserved')
slot = 128 * 1024**2
if s['state_slot_bytes'] != slot:
    raise ValueError('unexpected checkpoint slot size')
floor = 2 * 1024**3 + 128 * 1024**2 + 128 * 1024**2 + 6 * slot
if shutil.disk_usage(parent).free < floor:
    raise RuntimeError('insufficient mixed PPO data capacity')
# Historical whole-device peak is a sampled bound, not a reservation or an
# OOM guarantee; the complete mixed-corpus witness remains mandatory.
gpu_floor = 21428 * 1024**2 + 2668946 * 1792 + 4 * 1024**3
if s['gpu_memory_floor_bytes'] != gpu_floor:
    raise ValueError('mixed PPO GPU budget mismatch')
gpu = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                                   '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2
if gpu < gpu_floor:
    raise RuntimeError('insufficient initial GPU capacity')
sample_resources(ROOT, None, 16 * 1024**3, floor)
log.mkdir(mode=0o700)


def bounded_tree_bytes(root):
    if not root.exists():
        return 0
    total = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = pathlib.Path(folder) / name
            st = path.lstat()
            if path.is_symlink() or st.st_uid != 1013:
                raise RuntimeError('unexpected owner or symlink in output tree')
            if path.is_file():
                total += max(st.st_size, getattr(st, 'st_blocks', 0) * 512)
    return total


def sample(pid):
    receipt_path = out / 'receipt.json'
    complete = 0
    if receipt_path.exists():
        receipt = read(receipt_path)
        complete = len(receipt.get('checkpoints', []))
        if complete > 5:
            raise ValueError('too many checkpoint slots')
        for index, checkpoint in enumerate(receipt.get('checkpoints', []), 1):
            completed_iteration = index * 1000 if s['role'] == 'train' else index + (1 if s['role'] == 'document' else 0)
            expected = out / ('completed_' + str(completed_iteration) + '.private.pth')
            path = pathlib.Path(checkpoint['path'])
            if path != expected or path.resolve() != path or path.lstat().st_uid != 1013:
                raise ValueError('unexpected checkpoint path')
            if path.stat().st_size != checkpoint['bytes'] or not 0 < checkpoint['bytes'] <= slot:
                raise ValueError('checkpoint size or receipt mismatch')
    remaining = 6 - complete
    minimum = 2 * 1024**3 + 128 * 1024**2 + 128 * 1024**2 + remaining * slot
    value = sample_resources(ROOT, pid, 16 * 1024**3, minimum)
    value['all_log_bytes'] = bounded_tree_bytes(log) + bounded_tree_bytes(out / 'training_logs')
    metrics = out / 'metrics.private.jsonl'
    if metrics.exists():
        stat = metrics.lstat()
        if metrics.is_symlink() or stat.st_uid != 1013:
            raise RuntimeError('unsafe metrics file')
        value['all_log_bytes'] += max(stat.st_size, getattr(stat, 'st_blocks', 0) * 512)
    value['asset_bytes'] = bounded_tree_bytes(out / 'usd')
    if value['all_log_bytes'] > 128 * 1024**2:
        raise RuntimeError('mixed PPO log budget exceeded')
    if value['asset_bytes'] > 128 * 1024**2:
        raise RuntimeError('mixed PPO asset budget exceeded')
    return value


command = ['taskset', '-c', '0-3', '/usr/bin/env', '-u', 'PYTHONOPTIMIZE', '-u', 'CUDA_VISIBLE_DEVICES', '-u', 'PYTHONPATH',
           'CUDA_DEVICE_ORDER=PCI_BUS_ID', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPYCACHEPREFIX=' + str(cache), 'PYTHONIOENCODING=utf-8',
           'CUBLAS_WORKSPACE_CONFIG=:4096:8', 'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4',
           'bash', str(ROOT / 'scripts/run_tracker.sh'), str(ROOT / 'scripts/run_augmentation2526_mixed_ppo_v1.py'),
           '--request', str(request), '--trusted-request-sha256', a.trusted_request_sha256,
           '--headless', '--device', 'cuda:1']
proof = run_supervised(command, log / 'child.log', log / 'resources.jsonl', log / 'process_exit.json',
                       sample, 72 * 3600 if s['role'] == 'train' else 3600, 15, grace_seconds=30)
receipt_path = out / 'receipt.json'
receipt = read(receipt_path) if receipt_path.exists() else {}
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(request),
             receipt_sha256=sha(receipt_path) if receipt_path.exists() else None,
             resource_sha256=sha(log / 'resources.jsonl'), process_exit_sha256=sha(log / 'process_exit.json'),
             status=receipt.get('status'))
with (log / 'exit.json').open('x', encoding='utf-8') as f:
    json.dump(proof, f, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
expected = 'STAGE4_MIXED_TRAIN_ONE_STAGE_PPO_FINISHED' if s['role'] == 'train' else 'STAGE4_MIXED_REAL_PPO_RECOVERY_WITNESS_FINISHED'
if proof['supervision_passed'] is not True or proof['child_exit_code'] != 0 or receipt.get('status') != expected or receipt.get('output_writers_closed') is not True:
    raise RuntimeError('mixed PPO job did not close successfully')
