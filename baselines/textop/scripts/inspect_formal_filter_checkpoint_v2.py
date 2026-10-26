"""CPU-only checkpoint audit and public inference export; never delete or resume.

Full optimizer and per-motion sampling state stay in the original private file.
Only aggregate metadata and network/normalizer tensors enter the public export.
"""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import time

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN = 'filter_policy_paper69_solid_floor_20261026_v1'
PLAN_SHA = '153f8237a17928ffc3a28b7b00e366742a4ff0080e8fbb6bac87e57da8eb291d'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed', type=int, required=True)
    parser.add_argument('--audit-run', default='author_v2')
    args = parser.parse_args()
    completed = args.completed
    assert completed in [1000] + list(range(5000, 40001, 5000))
    assert os.getuid() == 1013
    assert re.fullmatch('[a-z0-9_]+', args.audit_run)
    guard = (ROOT / 'receipts/formal_checkpoint_audit.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1')
    checkpoint = ROOT / 'tracker_validation' / RUN / ('completed_{}.pt'.format(completed))
    receipt_path = checkpoint.parent / 'receipt.json'
    plan = ROOT / 'configs/filter-training-plan.json'
    assert sha(plan) == PLAN_SHA
    assert checkpoint.resolve() == checkpoint and not checkpoint.is_symlink()
    assert checkpoint.stat().st_uid == 1013 and time.time() - checkpoint.stat().st_mtime > 30
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt['status'] in ['RUNNING_FORMAL_FILTER_TRAINING', 'TEXTOP_FORMAL_FILTER_TRAINING_FINISHED']
    assert receipt['plan_sha256'] == PLAN_SHA and receipt['num_envs'] == 8192
    assert receipt['completed_learning_iterations'] >= completed and receipt['optimizer_updates'] >= completed * 20
    metadata = receipt['latest_checkpoint']
    assert metadata['completed_learning_iterations'] == completed, 'Audit latest stable checkpoint only'
    original_sha = sha(checkpoint)
    assert original_sha == metadata['sha256'] and checkpoint.stat().st_size == metadata['bytes']
    destination = ROOT / 'checkpoint_backups' / ('round_{}'.format(completed)) / args.audit_run
    assert not destination.exists(), 'Keep earlier audit/export; inspect instead of rerunning'
    assert shutil.disk_usage(ROOT).free > 3 * 1024**3
    destination.mkdir(parents=True)
    (destination / 'training_receipt_snapshot.json').write_bytes(receipt_bytes)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    # Own checkpoint is SHA-bound to the running trainer before this trusted legacy load.
    # Old Torch weights-only parser rejected protocol FLOAT; failure is retained.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    allowed = {'model_state_dict', 'optimizer_state_dict', 'iter', 'infos',
               'obs_norm_state_dict', 'privileged_obs_norm_state_dict'}
    assert set(state) == allowed, 'Unexpected fields require review before publication'
    info = state['infos']
    assert set(info) == {'completed_learning_iterations', 'actual_optimizer_updates',
                         'reference_fingerprint', 'diagnostic_only', 'recording'}
    assert state['iter'] == info['completed_learning_iterations'] == completed
    assert info['actual_optimizer_updates'] == completed * 20 and info['diagnostic_only'] is False
    assert info['reference_fingerprint'] == receipt['reference_fingerprint']
    tensor_count = [0]

    def finite(value):
        if torch.is_tensor(value):
            assert value.device.type == 'cpu' and torch.isfinite(value).all()
            tensor_count[0] += 1
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                finite(child)
        elif isinstance(value, float):
            assert __import__('math').isfinite(value)

    finite(state)
    steps = [int(x['step']) for x in state['optimizer_state_dict']['state'].values()]
    assert steps and min(steps) == max(steps) == completed * 20
    model = state['model_state_dict']
    assert tuple(model['actor.0.weight'].shape) == (2048, 557)
    assert tuple(model['critic.0.weight'].shape) == (2048, 557)
    assert tuple(model['actor.6.weight'].shape) == (29, 512)
    assert tuple(model['critic.6.weight'].shape) == (1, 512)
    recording = info['recording']
    for name in ['raw_failed', 'raw_success', 'failed_motion_count', 'success_motion_count']:
        assert tuple(recording[name].shape) == (8278,) and (recording[name] >= 0).all()
    export = {key: state[key] for key in ['model_state_dict', 'obs_norm_state_dict',
                                        'privileged_obs_norm_state_dict', 'iter']}
    export['infos'] = {'completed_learning_iterations': completed,
                       'actual_optimizer_updates': completed * 20,
                       'reference_fingerprint': info['reference_fingerprint'],
                       'diagnostic_only': False, 'inference_only_export': True,
                       'full_resume_requires_private_checkpoint': True}
    public = destination / ('TextOp_FilterPolicy29D_round{}_inference.pt'.format(completed))
    with public.open('xb') as stream:
        torch.save(export, stream)
    restored = torch.load(public, map_location='cpu', weights_only=False)
    checked = 0
    for key in ['model_state_dict', 'obs_norm_state_dict', 'privileged_obs_norm_state_dict']:
        assert restored[key].keys() == state[key].keys()
        for name in state[key]:
            assert torch.equal(restored[key][name], state[key][name])
            checked += 1
    assert 'optimizer_state_dict' not in restored and 'recording' not in restored['infos']
    assert sha(checkpoint) == original_sha and not torch.cuda.is_initialized()
    proof = {'status': 'FORMAL_CHECKPOINT_CPU_AUDITED_PUBLIC_INFERENCE_EXPORTED',
             'run': RUN, 'completed_learning_iterations': completed,
             'actual_optimizer_updates': info['actual_optimizer_updates'],
             'optimizer_step_min': min(steps), 'optimizer_step_max': max(steps),
             'all_finite_tensor_count': tensor_count[0], 'exact_public_tensors_checked': checked,
             'full_checkpoint': {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size, 'sha256': original_sha},
             'public_inference': {'path': str(public), 'bytes': public.stat().st_size, 'sha256': sha(public)},
             'training_receipt_snapshot_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
             'reference_fingerprint': info['reference_fingerprint'], 'plan_sha256': PLAN_SHA,
             'cpu_affinity': sorted(os.sched_getaffinity(0)), 'torch_cuda_initialized': False,
             'torch_version':torch.__version__, 'checkpoint_read_mode':'trusted_sha_verified_cpu_legacy_loader',
             'sdk_runs': 0, 'new_optimizer_updates': 0, 'data_deleted': False,
             'public_contains_per_motion_statistics': False, 'public_contains_optimizer': False,
             'full_resume_checkpoint_remains_private': True,
             'physical_filter_executed': False, 'full_training_finished': completed == 40000,
             'tracking_convergence_claimed': False, 'source_sha256': sha(pathlib.Path(__file__)),
             'sample_unix': time.time()}
    with (destination / 'audit.json').open('x') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
