"""Resume the pinned trainer with verified state and an unambiguous update count.

Does not change the reward, observations, network, or PPO update. The pinned
RSL-RL loader restores the last completed iteration as its next loop index and
does not restore the algorithm's adaptive learning-rate scalar. This wrapper
checks restored state, starts at the next index, and synchronizes that scalar
with the restored optimizer. Simulator and random-generator state are reset.
"""
from pathlib import Path
import hashlib, json, os, runpy, sys
import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

original_load = OnPolicyRunner.load

def equal_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and actual.shape == expected.shape
                and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equal_state(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(equal_state(a, b) for a, b in zip(actual, expected))
    return actual == expected

def checked_load(self, path, load_optimizer=True):
    assert load_optimizer, 'Continuation requires the saved optimizer state'
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest == os.environ['PHYGILE_RESUME_PARENT_SHA256'], 'Unexpected parent checkpoint'
    checkpoint = torch.load(p, map_location='cpu', weights_only=False)
    info = original_load(self, path, load_optimizer=load_optimizer)
    assert equal_state(self.alg.policy.state_dict(), checkpoint['model_state_dict'])
    assert equal_state(self.alg.optimizer.state_dict(), checkpoint['optimizer_state_dict'])
    if self.empirical_normalization:
        assert equal_state(self.obs_normalizer.state_dict(), checkpoint['obs_norm_state_dict'])
        assert equal_state(self.privileged_obs_normalizer.state_dict(), checkpoint['privileged_obs_norm_state_dict'])
    rates = [float(g['lr']) for g in self.alg.optimizer.param_groups]
    assert rates and len(set(rates)) == 1, 'This continuation only supports a single shared optimizer learning rate'
    previous_scalar = float(self.alg.learning_rate)
    self.alg.learning_rate = rates[0]
    self.current_learning_iteration = int(checkpoint['iter']) + 1
    record = {
        'parent_checkpoint': str(p.resolve()), 'parent_sha256': digest,
        'parent_last_completed_index': int(checkpoint['iter']),
        'parent_optimizer_updates': int(checkpoint['iter']) + 1,
        'first_new_update_index': self.current_learning_iteration,
        'policy_optimizer_and_normalizers_match_parent': True,
        'optimizer_learning_rates': rates,
        'algorithm_lr_before_sync': previous_scalar,
        'algorithm_lr_after_sync': self.alg.learning_rate,
        'additional_updates_requested': int(self.cfg['max_iterations']),
        'active_experts': self.alg.policy._max_active_experts,
        'exact_trajectory_resume': False,
        'limitation': 'Simulator, RNG, rollout buffer and sampler state are restarted; this is checkpoint continuation, not bit-exact continuation.'
    }
    (Path(self.log_dir) / 'resume_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print('PHYGILE_VERIFIED_RESUME', json.dumps(record), flush=True)
    return info

OnPolicyRunner.load = checked_load
train = Path(__file__).resolve().parents[1] / 'upstream/scripts/rsl_rl/train.py'
sys.path.insert(0, str(train.parent))
sys.argv[0] = str(train)
runpy.run_path(str(train), run_name='__main__')
