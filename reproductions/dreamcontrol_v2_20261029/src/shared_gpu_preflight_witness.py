"""Read-only shared-GPU resource preflight; never initializes CUDA."""
import hashlib
import json
from pathlib import Path

import torch
from train_robot_prior import resource_preflight

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was already initialized')
    resource_preflight(torch.device('cuda:1'), HERE, checkpoint_size_estimate=1024**3,
                       max_steps=1, checkpoint_every=1, allow_shared_gpu=True, gpu_memory_fraction=.1)
    if torch.cuda.is_initialized():
        raise RuntimeError('preflight unexpectedly initialized CUDA')
    result = {'device': 'cuda:1', 'allow_shared_gpu': True, 'allocator_fraction': .1,
              'cuda_initialized': False, 'entry_sha256': sha(HERE / 'train_robot_prior.py'),
              'test_sha256': sha(HERE / 'test_train_robot_prior.py'),
              'script_sha256': sha(Path(__file__)), 'resource_preflight_passed_at_check': True,
              'training_ready': False, 'limitations': ['Read-only point-in-time check cannot reserve GPU or disk.',
              'Allocator fraction caps PyTorch caching allocator only, not all CUDA allocations.',
              'Model memory is not measured and quality/source acceptance remains blocked.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
