"""Read-only CPU reload of the real pilot checkpoint, optimizer and official model."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from witness_runtime import BASE, configuration

p = argparse.ArgumentParser()
p.add_argument('--runtime-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
torch.set_num_threads(4)
torch.manual_seed(42)
from reactivebfm.data.datasets import get_dataset_loader
from reactivebfm.model.motion_planner.factory import create_model_and_diffusion_smooth
run = a.runtime_root / 'runs/pilot_v1'
spec = json.loads((BASE / 'configs/env_spec.json').read_text())
args = configuration(spec)
loader = get_dataset_loader('kungfu_g1_36dim', batch_size=1,
    data_dir=str(a.runtime_root / 'data/kungfu_g1_36dim'), split='test',
    fixed_len=175, pred_len=50, device='cpu', collate_mode='self_rollout',
    num_workers=0, shuffle=False, deterministic=True)
model, _ = create_model_and_diffusion_smooth(args, loader)
state = torch.load(run / 'checkpoints/model000000100.pt', map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(state, strict=False)
if unexpected or any(not k.startswith('clip_model.') for k in missing):
    raise RuntimeError('Unexpected state-dict incompatibility')
h = hashlib.sha256()
for name, value in model.named_parameters():
    if value.requires_grad:
        h.update(name.encode()); h.update(value.detach().numpy().tobytes())
receipt = json.loads((run / 'run_receipt.json').read_text())
assert h.hexdigest() == receipt['final_trainable_sha256']
opt = torch.load(run / 'checkpoints/opt000000100.pt', map_location='cpu', weights_only=True)
steps = sorted(set(int(v['step']) for v in opt['state'].values()))
assert steps == [101], steps
model.eval()
motion, cond = next(iter(loader))
y = {'prefix': motion[..., :25], 'mask': torch.ones(1, 1, 1, 50, dtype=torch.bool), 'text': cond['y']['text']}
with torch.no_grad():
    prediction = model(torch.randn_like(motion[..., 25:75]), torch.tensor([5], dtype=torch.long), y=y)
assert list(prediction.shape) == [1, 36, 1, 50] and torch.isfinite(prediction).all()
result = {'status': 'CHECKPOINT_RELOAD_AND_FORWARD_PASSED', 'device': 'cpu',
    'trainable_sha256': h.hexdigest(), 'matches_final_training_hash': True,
    'optimizer_state_count': len(opt['state']), 'optimizer_step_values': steps,
    'missing_frozen_text_keys': len(missing), 'unexpected_keys': unexpected,
    'prediction_shape': list(prediction.shape), 'finite_prediction': True,
    'scope': 'checkpoint usability only; no motion-quality or physical-control evaluation',
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
