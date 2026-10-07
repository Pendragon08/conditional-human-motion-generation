"""Preserve executed v1; create a resource-witness version before dispatch."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'scripts/r3_world_ppo_control_isaac_20261007.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='fa85dcbdfdb2f8e5a45dd04796343d106ee042e4cacd6d17ba50b493b93e99f9'
s=p.read_text(encoding='utf-8')
s=s.replace("choices=['smoke','formal']","choices=['smoke','capacity','formal']",1)
s=s.replace("else:assert a.updates==2 and a.num_envs==256", "elif a.mode=='capacity':\n        assert a.updates==2 and a.num_envs==8192 and a.arm in ['P-history','W-correct']\n    else:assert a.updates==2 and a.num_envs==256",1)
s=s.replace("'diagnostic_only':a.mode=='smoke'", "'diagnostic_only':a.mode!='formal'")
s=s.replace("if a.mode=='smoke' else 0", "if a.mode!='formal' else 0")
s=s.replace("'stage':'training' if a.mode=='formal' else 'smoke'", "'stage':'training' if a.mode=='formal' else 'smoke'")
# Capacity keeps full histories in memory (the real worst case), but saves only
# metadata/fixtures rather than another complete 8192 raw rollout to disk.
needle="algorithm.storage.clear()\n    torch.manual_seed(a.seed+9000)"
replacement="""if a.mode=='capacity':
            torch.save({'pre_action_values':oldvalue[:4].clone(), 'timeout':timeout[:4].clone(),
                'done':done_fixture[:4].clone(), 'stored_reward':algorithm.storage.rewards[0,:4].clone(),
                'actual_reward':torch.zeros(4,device=a.device)}, a.output/'timeout_fixture.pt')
        algorithm.storage.clear()
    torch.manual_seed(a.seed+9000)"""
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace("losses=algorithm.update()", """losses=algorithm.update()
            if a.mode=='capacity':
                gradients=[p.grad for p in policy.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                assert all(p.grad is None for p in world.parameters())
            """,1)
s=s.replace("rollout_started=time.time();wm_frozen=", "world.zero_grad(set_to_none=True)\n            rollout_started=time.time();wm_frozen=",1)
target=ROOT/'scripts/r3_world_ppo_control_capacity_v2_20261007.py'
target.write_text(s,encoding='utf-8',newline='\n')
print('CAPACITY_SOURCE_CREATED',hashlib.sha256(target.read_bytes()).hexdigest())
