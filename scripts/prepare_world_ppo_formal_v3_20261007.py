"""Keep executed v1/v2; v3 adds saved audit and read-only training metrics."""
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parents[1]
src=root/'scripts/r3_world_ppo_control_capacity_v2_20261007.py'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='459075bdf79ae66255b08033f4920812407b259e12026adfc00e8dc7d950945b'
s=src.read_text(encoding='utf-8')
s=s.replace("from isaaclab_tasks.utils import parse_env_cfg", "from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry",1)
needle="cfg=parse_env_cfg('Isaac-Motion-Tracking-Kick-v0',device=a.device,num_envs=a.num_envs);cfg.seed=a.seed"
replacement="""agentcfg=load_cfg_from_registry('Isaac-Motion-Tracking-Kick-v0','rsl_rl_cfg_entry_point')
    assert agentcfg.clip_actions is None,'Raw env is equivalent only when original wrapper clipping is None'
    cfg=parse_env_cfg('Isaac-Motion-Tracking-Kick-v0',device=a.device,num_envs=a.num_envs);cfg.seed=a.seed"""
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace("if a.mode=='capacity':\n                gradients=", "if a.mode in ['capacity','formal']:\n                gradients=",1)
s=s.replace("auxaudit=[];handoff_kl=0.;matched=0;eligible=0", """world_after_PPO_hash=tensor_hash(world.state_dict())
            world_before_PPO_hash=tensor_hash(wm_frozen)
            assert world_after_PPO_hash==world_before_PPO_hash
            if a.mode=='formal' and (update==0 or (update+1)%50==0):
                torch.save({'world_before_and_after_PPO':wm_frozen,
                    'world_hash':world_before_PPO_hash,'world_version':world_version,
                    'world_gradients_during_PPO':[p.grad for p in world.parameters()]},
                    a.output/f'world_PPO_boundary_update{update+1:04d}.pt')
            auxaudit=[];handoff_kl=0.;matched=0;eligible=0""",1)
s=s.replace("'mean_actual_reward':reward_sum/24", """'training_any_reset_success_fraction_cumulative':float((raw.n_successes>0).float().mean()),
                'training_success_events_cumulative':int(raw.n_successes.sum()),
                'world_before_PPO_hash':world_before_PPO_hash,'world_after_PPO_hash':world_after_PPO_hash,
                'mean_actual_reward':reward_sum/24""",1)
s=s.replace("writer.add_scalar('World/encoder_handoff_KL',handoff_kl,update+1)", """writer.add_scalar('World/encoder_handoff_KL',handoff_kl,update+1)
            writer.add_scalar('Train/cumulative_any_reset_success_fraction',row['training_any_reset_success_fraction_cumulative'],update+1)
            for idx,v in enumerate(auxaudit):writer.add_scalar('World/aux_prediction_loss',v['prediction_loss'],update+1)""",1)
param='''{'PPO_trainable':sum(p.numel() for p in policy.parameters() if p.requires_grad),
                'aux_trainable':sum(p.numel() for p in world.parameters()) if a.arm.startswith('W-') else 0,
                'deployed_actor':sum(p.numel() for p in policy.actor.parameters())+policy.std.numel()+
                 (sum(p.numel() for p in policy.history_encoder.parameters()) if a.arm in ['P-cap','P-history'] else
                  sum(p.numel() for p in world.encoder.parameters()) if a.arm.startswith('W-') else 0)}'''
s=s.replace("'innovation_validated':False,'wall_s':time.time()-started,'audit':audit", "'parameter_counts':"+param+",\n            'innovation_validated':False,'wall_s':time.time()-started,'audit':audit",1)
target=root/'scripts/r3_world_ppo_control_formal_v3_20261007.py'
assert not target.exists();target.write_text(s,encoding='utf-8',newline='\n')
print('FORMAL_V3_AUDIT_SOURCE_CREATED',hashlib.sha256(target.read_bytes()).hexdigest())
