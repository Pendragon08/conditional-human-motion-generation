"""Read-only comparison of this experiment's trusted 500/2000-update checkpoints."""
from pathlib import Path
import hashlib
import json
import torch
import yaml

root = Path(__file__).resolve().parents[1]
out = root/'results/continuation_2000'
assert json.loads((out/'pipeline.json').read_text())['status'] == 'COMPLETED'
parents = list((root/'logs/rsl_rl/phygile_moe_pilot').glob('*train20_nostate_seed42/model_499.pt'))
finals = list((root/'logs/rsl_rl/phygile_moe_pilot').glob('*extended2000_seed42/model_1999.pt'))
assert len(parents) == len(finals) == 1
paths = [parents[0], finals[0]]
checkpoints = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
assert [x['iter'] for x in checkpoints] == [499, 1999]
states = [c['model_state_dict'] for c in checkpoints]
assert states[0].keys() == states[1].keys()
assert all(torch.isfinite(t).all() for t in states[1].values() if isinstance(t, torch.Tensor))
changes = {}
for index in range(12):
    keys = [key for key in states[0] if key.startswith(f'experts.{index}.')]
    assert keys
    changes[str(index)] = {'tensors':len(keys), 'changed_tensors':sum(not torch.equal(states[0][key], states[1][key]) for key in keys)}
assert changes['0']['changed_tensors'] > 0
assert all(changes[str(i)]['changed_tensors'] == 0 for i in range(1, 12))

agent = yaml.safe_load((finals[0].parent/'params/agent.yaml').read_text())
expected_step_difference = 1500 * agent['algorithm']['num_learning_epochs'] * agent['algorithm']['num_mini_batches']
optimizer_states = [c['optimizer_state_dict']['state'] for c in checkpoints]
assert optimizer_states[0].keys() == optimizer_states[1].keys()
step_differences = []
parent_steps, final_steps = [], []
for key, value in optimizer_states[0].items():
    if 'step' not in value:
        continue
    before = int(value['step']); after = int(optimizer_states[1][key]['step'])
    parent_steps.append(before); final_steps.append(after)
    step_differences.append(after-before)
assert step_differences and set(step_differences) == {expected_step_difference}
assert set(parent_steps) == {10000} and set(final_steps) == {40000}

result = {'status':'PASSED', 'parent_sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest(),
    'final_sha256':hashlib.sha256(paths[1].read_bytes()).hexdigest(), 'all_final_model_tensors_finite':True,
    'expert_parameter_changes':changes, 'optimizer_tensors_with_step':len(step_differences),
    'optimizer_step_differences':sorted(set(step_differences)),
    'parent_adam_steps':sorted(set(parent_steps)), 'final_adam_steps':sorted(set(final_steps)),
    'expected_step_difference':expected_step_difference,
    'basis':'1500 additional PPO learning iterations times saved epochs times saved minibatches; legacy optimizer_updates fields denote PPO iterations',
    'note':'Only expert 0 changed; no evidence of full curriculum or multi-expert specialization.'}
(out/'state_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
