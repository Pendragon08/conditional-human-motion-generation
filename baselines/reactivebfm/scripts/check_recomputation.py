"""Real-data FP32 gradient parity for supplementary DiT block recomputation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.utils.checkpoint import checkpoint
from witness_runtime import BASE, configuration
from reactivebfm.data.datasets import get_dataset_loader
from reactivebfm.model.motion_planner.factory import create_model_and_diffusion_smooth
from reactivebfm.model.motion_planner.architectures.dit import DiTBlock

p = argparse.ArgumentParser(); p.add_argument('--runtime-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True); a = p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.25, 0)
torch.manual_seed(42)
spec = json.loads((BASE / 'configs/env_spec.json').read_text()); args = configuration(spec)
loader = get_dataset_loader('kungfu_g1_36dim', batch_size=2,
    data_dir=str(a.runtime_root / 'data/kungfu_g1_36dim'), split='train', fixed_len=175,
    pred_len=50, device='cpu', collate_mode='self_rollout', num_workers=0, shuffle=False, deterministic=True)
motion, cond = next(iter(loader)); motion = motion.cuda()
model, _ = create_model_and_diffusion_smooth(args, loader); model.cuda().train()
# Exercise nonzero backbone gradients rather than the initial zero-gated DiT.
perturbed = 0
with torch.no_grad():
    for value in model.parameters():
        if value.requires_grad and not torch.count_nonzero(value):
            value.add_(torch.randn_like(value) * .01); perturbed += 1
torch.manual_seed(99); noise = torch.randn_like(motion[..., 25:75])
def gradients():
    model.zero_grad(set_to_none=True); torch.manual_seed(123)
    y = {'prefix': motion[..., :25], 'mask': torch.ones(2, 1, 1, 50, device='cuda', dtype=torch.bool), 'text': cond['y']['text']}
    pred = model(noise, torch.tensor([5, 5], device='cuda'), y=y)
    loss = (pred - motion[..., 25:75]).square().mean(); loss.backward()
    return float(loss.detach()), {n: p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
loss0, grads0 = gradients(); original = DiTBlock.forward
def memory_forward(self, *args, **kwargs):
    return checkpoint(original, self, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
DiTBlock.forward = memory_forward
loss1, grads1 = gradients(); DiTBlock.forward = original
assert grads0.keys() == grads1.keys()
nonzero = sum(bool(torch.count_nonzero(g)) for g in grads0.values())
assert nonzero >= 200
errors = {n: float((grads0[n] - grads1[n]).abs().max()) for n in grads0}
assert all(torch.allclose(grads0[n], grads1[n], atol=1e-6, rtol=1e-5) for n in grads0)
assert abs(loss0 - loss1) < 1e-6
result = {'status': 'RECOMPUTATION_GRADIENT_PARITY_PASSED', 'seed': 42,
    'reference_loss': loss0, 'recomputed_loss': loss1, 'gradient_tensors_compared': len(errors),
    'maximum_absolute_gradient_error': max(errors.values()), 'atol': 1e-6, 'rtol': 1e-5,
    'nonzero_gradient_tensors': nonzero, 'zero_initialized_parameters_perturbed': perturbed,
    'scope': 'One real-data FP32 batch with deterministic nonzero gate/output perturbations; confirms recomputation hook parity, not BF16/whole-training equivalence.'}
a.output.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result))
