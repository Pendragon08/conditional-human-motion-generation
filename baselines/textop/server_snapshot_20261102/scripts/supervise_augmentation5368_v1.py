"""Exclusive bounded supervisor for the authorized TRAIN augmentation job."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess

from filtered15000_vae_completion import ROOT, owned, read, sha
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = owned(args.request)
    s = read(request)
    assert os.getuid() == 1013 and s['schema'] == 'textop.augmentation5368_job.v1'
    assert s['authorized'] is True and sha(__file__) == s['supervisor_sha256']
    assert s['run'] == 'user_augmentation5368_' + s['role'] + '_v1'
    assert s['role'] in ('author', 'document', 'production')
    for name, digest in s['bound_sources'].items():
        assert sha(owned(ROOT / name)) == digest
    assert sha(owned(ROOT / 'scripts/run_augmentation5368_v1.py')) == s['executor_sha256']
    lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # The immediately preceding PPO was intentionally interrupted by the user.
    ppo = 'user_step4_train_only_ppo5000_natural_train_v2'
    receipt = owned(ROOT / 'step34_numeric_v1' / ppo / 'receipt.json')
    assert sha(receipt) == s['stopped_ppo_receipt_sha256']
    stopped = read(receipt)
    assert stopped['status'] == 'STAGE4_ORIGINAL_TRAIN_PPO_FAILED_PARTIAL_RETAINED'
    assert stopped['error'] == "RuntimeError('Owned Stage4 PPO interrupted with signal 2')"
    assert read(ROOT / 'receipts/user_stop_natural_ppo_v2_20261030_sent.json')['signals_sent'] == 1
    outer = ROOT / 'receipts' / ('step4_train_only_' + ppo + '_outer_exit.json')
    assert read(outer)['actual_returncode'] == 1
    from augmentation5368_verified_inputs_v1 import accepted_inputs
    actual = accepted_inputs()
    clips = actual['plan']['clips'] if s['role'] == 'production' else actual['plan']['clips'][:1]
    expected = sum(c['target_pose_frames'] * 152 + 4096 for c in clips) + 128 * 1024**2
    assert s['output_upper_bytes'] == expected
    floor = s['reserve_bytes'] + s['log_allowance_bytes'] + expected + s['future_pipeline_upper_bytes']
    assert shutil.disk_usage(ROOT).free >= floor
    pre = sample_resources(ROOT, None, s['minimum_ram_bytes'], floor)
    gpu = int(subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2
    assert gpu >= s['minimum_gpu_bytes']
    parent = owned(ROOT / 'step34_20261029')
    assert not (parent / s['run']).exists()
    log = parent / (s['run'] + '_supervision')
    log.mkdir(mode=0o700, exist_ok=False)
    def sample(pid):
        value = sample_resources(ROOT, pid, s['minimum_ram_bytes'], floor)
        value['direct_log_bytes'] = (log / 'child.log').stat().st_size if (log / 'child.log').exists() else 0
        assert value['direct_log_bytes'] <= s['log_allowance_bytes']
        return value
    command = ['taskset', '-c', '0-3', '/usr/bin/env', 'CUDA_VISIBLE_DEVICES=1',
        'PYTHONIOENCODING=utf-8', 'PYTHONUNBUFFERED=1', 'OMP_NUM_THREADS=4',
        'OPENBLAS_NUM_THREADS=4', str(ROOT / 'env/bin/python'),
        str(ROOT / 'scripts/run_augmentation5368_v1.py'), '--request', str(request)]
    proof = run_supervised(command, log / 'child.log', log / 'resources.jsonl',
                           log / 'process_exit.json', sample, s['timeout_seconds'], 15)
    rp = parent / s['run'] / 'receipt.json'
    rec = read(rp) if rp.exists() else {}
    proof.update(run=s['run'], role=s['role'], request_sha256=sha(request),
                 source_sha256=sha(__file__), receipt_sha256=sha(rp) if rp.exists() else None,
                 resource_sha256=sha(log / 'resources.jsonl'),
                 process_exit_sha256=sha(log / 'process_exit.json'),
                 preflight_ram_bytes=pre['available_host_ram_bytes'], preflight_gpu_free_bytes=gpu,
                 status=rec.get('status'))
    with (log / 'exit.json').open('x') as stream:
        json.dump(proof, stream, indent=2, allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    assert proof['supervision_passed'] and proof['child_exit_code'] == 0
    assert rec['status'] == 'TRAIN_AUGMENTATION_FINISHED' and rec['output_writers_closed'] is True


if __name__ == '__main__':
    main()
