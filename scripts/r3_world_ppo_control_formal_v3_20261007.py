"""Real-rollout PPO with causal residual features. Smoke is never a final score.

No imagined transitions/rewards, new IK, PD/reward changes, or future actor input.
"""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['smoke','capacity','formal'],required=True)
p.add_argument('--arm',choices=['P0','P-cap','P-history','W-correct','W-shuffled-original',
                                'W-shuffled-initmatched','W-pretrained-frozen'],required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--num_envs',type=int,default=8192)
p.add_argument('--updates',type=int,default=500)
p.add_argument('--aux-updates',type=int,default=1)
p.add_argument('--aux-batch',type=int,default=1024)
p.add_argument('--monitor-prefix',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();sys.argv=[sys.argv[0]]
launcher=AppLauncher(a);app=launcher.app

import faulthandler
import hashlib
import json
import time
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder
from isaaclab.utils.math import quat_apply_inverse
from rsl_rl.modules import ActorCritic,EmpiricalNormalization
from rsl_rl.algorithms import PPO
from torch.utils.tensorboard import SummaryWriter
from r3_i1_training_telemetry import SampledTrainingTelemetry
from r3_world_ppo_core import WorldModel,HistoryBuffer,AugmentedActorCritic,INPUT,SCHEMA_HASH
from r3_world_ppo_residual_core import ResidualWorldModel
from r3_world_ppo_training_core_20261007 import (restore_original_adam,tensor_hash,
    make_windows,auxiliary_step,feature_observation,gaussian_kl)

ROOT=Path(__file__).resolve().parents[1]
CKPT=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt')
EXPECTED='e3c50fb6f36f59f89d1b4e788b1ba74cf1ff0cdde3dceff34a9d7765235f65c0'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    assert not a.output.exists(),'Preserve prior invocation; no blind retries'
    if a.mode=='formal':
        assert a.num_envs==8192 and a.updates==500 and a.seed in [42,43,44]
        gate=json.loads((ROOT/'artifacts/monitoring/r3_world_residual_ppo_local_view_gate_20261007.json').read_text())
        assert gate['user_confirmed_local_stream'] and gate['source']=='actual_isaaclab_simulation'
        assert json.loads((ROOT/'artifacts/world_confirmation_20261007/confirmation_v2/evaluate_seed42_v1/summary.json').read_text())['formal_prediction_gate_passed']
        for seed in [43,44]:
            assert json.loads((ROOT/f'artifacts/world_confirmation_20261007/confirmation_v2/evaluate_seed{seed}_v1/summary.json').read_text())['formal_prediction_gate_passed']
    elif a.mode=='capacity':
        assert a.updates==2 and a.num_envs==8192 and a.arm in ['P-history','W-correct']
    else:assert a.updates==2 and a.num_envs==256
    assert a.aux_updates==1 and a.aux_batch==1024  # lock before any control score
    a.output.mkdir(parents=True);faulthandler.enable();faulthandler.dump_traceback_later(240,repeat=True)
    started=time.time();torch.manual_seed(a.seed);np.random.seed(a.seed)
    assert sha(CKPT)==EXPECTED
    ckpt=torch.load(CKPT,map_location=a.device,weights_only=False)
    original=ActorCritic(626,626,29,actor_hidden_dims=[512,256,256],critic_hidden_dims=[512,256,256],activation='elu').to(a.device)
    original.load_state_dict(ckpt['model_state_dict']);original.eval()
    norm=EmpiricalNormalization(626).to(a.device);norm.load_state_dict(ckpt['obs_norm_state_dict']);norm.eval()
    assert int(ckpt['privileged_obs_norm_state_dict']['count'])==0
    norm_before={k:v.clone() for k,v in norm.state_dict().items()}
    # Same random capacity/history encoder and same development-only statistics.
    fit=ROOT/f'artifacts/world_residual_20261007/fit_seed{a.seed}_v1'
    correct=torch.load(fit/'residual_correct.pt',map_location=a.device,weights_only=False)
    selected=correct
    if a.arm=='W-shuffled-original':
        selected=torch.load(fit/'residual_shuffled.pt',map_location=a.device,weights_only=False)
    coefficients={k:selected['state_dict']['ridge_'+str(k)] for k in [1,5,10]}
    world=ResidualWorldModel(coefficients).to(a.device);world.load_state_dict(selected['state_dict'])
    assert selected['schema_hash']==SCHEMA_HASH
    torch.manual_seed(a.seed);template=WorldModel().to(a.device)
    policy=AugmentedActorCritic(a.arm,ckpt['model_state_dict'],template.encoder.state_dict()).to(a.device)
    policy.input_mean.copy_(correct['state_dict']['input_mean']);policy.input_std.copy_(correct['state_dict']['input_std'])
    algorithm=PPO(policy,device=a.device,num_learning_epochs=5,num_mini_batches=4,clip_param=.2,
                  gamma=.99,lam=.95,value_loss_coef=1.,entropy_coef=.008,learning_rate=.001,
                  max_grad_norm=1.,use_clipped_value_loss=True,schedule='adaptive',desired_kl=.01)
    restored=restore_original_adam(algorithm,original,ckpt)
    dim={'P0':626,'P-cap':718,'P-history':1546}.get(a.arm,754)
    algorithm.init_storage('rl',a.num_envs,24,[dim],[dim],[29])
    auxiliary=torch.optim.Adam(world.parameters(),lr=.001)
    ridge_norm_keys=[k for k in world.state_dict() if k.startswith('ridge_') or k.endswith('_mean') or k.endswith('_std')]
    ridge_norm={k:world.state_dict()[k].clone() for k in ridge_norm_keys}
    agentcfg=load_cfg_from_registry('Isaac-Motion-Tracking-Kick-v0','rsl_rl_cfg_entry_point')
    assert agentcfg.clip_actions is None,'Raw env is equivalent only when original wrapper clipping is None'
    cfg=parse_env_cfg('Isaac-Motion-Tracking-Kick-v0',device=a.device,num_envs=a.num_envs);cfg.seed=a.seed
    env=SampledTrainingTelemetry(gym.make('Isaac-Motion-Tracking-Kick-v0',cfg=cfg),'127.0.0.1:40123')
    raw=env.unwrapped;robot=raw.scene['robot'];lib=raw.motion_lib
    assert 'critic' not in raw.observation_manager.active_terms and raw.step_dt==.02
    refs=np.load(ROOT/'artifacts/sim2sim_mujoco/interfaces/Kick_references.npz')
    ref_error=float(np.max(np.abs(lib.dof_pos.cpu().numpy()-refs['dof_pos'])));assert ref_error<1e-6
    assert len(lib._motion_data_list)==66
    jid=[robot.joint_names.index(x) for x in JointNamesOrder]
    feet=[robot.body_names.index(x) for x in ['left_ankle_roll_link','right_ankle_roll_link']]
    sensor=raw.scene['contact_forces'];cid=[sensor.body_names.index(robot.body_names[x]) for x in feet]
    obs,_=env.reset(seed=a.seed);obs=obs['policy'];history=HistoryBuffer(a.num_envs,a.device);history.append(obs)
    torch.manual_seed(a.seed+9000)
    with torch.no_grad():
        base=norm(obs);reference_mean=original.act_inference(base);reference_value=original.evaluate(base)
        augmented=feature_observation(a.arm,norm,obs,history.value,world)
        policy.act(augmented);error_mean=float((policy.action_mean-reference_mean).abs().max())
        error_value=float((policy.evaluate(augmented)-reference_value).abs().max())
        error_std=float((policy.action_std-original.std.expand(a.num_envs,-1)).abs().max())
        original.act(base);reference_logp=original.get_actions_log_prob(policy.action_mean)
        error_logp=float((policy.get_actions_log_prob(policy.action_mean)-reference_logp).abs().max())
        assert error_mean<1e-5 and error_value<1e-5 and error_std==0 and error_logp<1e-4
        # Isolated system-semantic fixture, not another environment rollout.
        algorithm.act(augmented,augmented)
        oldvalue=algorithm.transition.values.clone()
        timeout=torch.zeros(a.num_envs,device=a.device);timeout[0]=1
        done_fixture=timeout.bool();done_fixture[1]=True
        algorithm.process_env_step(torch.zeros(a.num_envs,device=a.device),done_fixture,{'time_outs':timeout})
        assert torch.allclose(algorithm.storage.rewards[0,:,0],.99*oldvalue[:,0]*timeout,atol=1e-6,rtol=1e-6)
        if a.mode=='capacity':
            torch.save({'pre_action_values':oldvalue[:4].clone(), 'timeout':timeout[:4].clone(),
                'done':done_fixture[:4].clone(), 'stored_reward':algorithm.storage.rewards[0,:4].clone(),
                'actual_reward':torch.zeros(4,device=a.device)}, a.output/'timeout_fixture.pt')
        algorithm.storage.clear()
    torch.manual_seed(a.seed+9000)
    atomic(ROOT/'artifacts/monitoring/r3_active_dreamcontrol_training.json',{
        'task_id':'Isaac-Motion-Tracking-Kick-v0','stage':'training' if a.mode=='formal' else 'smoke',
        'is_formal_training':a.mode=='formal','source':'actual_isaaclab_simulation','budget':a.updates,
        'arm':a.arm,'output':str(a.output),'seed':a.seed,'num_envs':a.num_envs,
        'queue_state':str(a.monitor_prefix.with_suffix('.state')),'log':str(a.monitor_prefix.with_suffix('.log'))})
    def labels():
        rel=robot.data.body_pos_w[:,feet]-robot.data.root_pos_w[:,None,:]
        q=robot.data.root_quat_w[:,None,:].expand(-1,2,-1)
        relb=quat_apply_inverse(q.reshape(-1,4),rel.reshape(-1,3)).reshape(a.num_envs,6)
        contact=(sensor.data.net_forces_w[:,cid].norm(dim=-1)>5).float()
        return torch.cat([robot.data.joint_pos[:,jid],robot.data.projected_gravity_b,
            robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,relb,
            robot.data.body_lin_vel_w[:,feet,:2].flatten(1),
            (robot.data.root_pos_w[:,2]-raw.scene.env_origins[:,2])[:,None],robot.data.root_vel_w[:,2:3],contact],-1)
    episode=torch.zeros(a.num_envs,dtype=torch.int32,device=a.device);total_resets=0;aux_count=0;world_version=0
    writer=SummaryWriter(str(a.output/'tensorboard'));audit=[]
    policy_before=tensor_hash(policy.state_dict());world_initial=tensor_hash(world.state_dict())
    if a.mode=='smoke':
        torch.save({'policy':policy.state_dict(),'optimizer':algorithm.optimizer.state_dict(),
                    'original_optimizer':ckpt['optimizer_state_dict'],
                    'world_model':world.state_dict(),'actor_obs_normalizer':norm.state_dict()},
                   a.output/'initial_restoration_audit.pt')
    def save_checkpoint(update):
        torch.save({'policy':policy.state_dict(),'optimizer':algorithm.optimizer.state_dict(),
            'world_model':world.state_dict() if a.arm.startswith('W-') else None,
            'aux_optimizer':auxiliary.state_dict() if a.arm.startswith('W-') else None,
            'actor_obs_normalizer':norm.state_dict(),'learning_rate':algorithm.learning_rate,
            'completed_updates':update,'world_version':world_version,'arm':a.arm,'seed':a.seed,
            'schema_hash':SCHEMA_HASH,'original_checkpoint_sha256':EXPECTED,
            'no_extra_IK':True,'diagnostic_only':a.mode!='formal'},a.output/f'model_update{update:04d}.pt')
    try:
        for update in range(a.updates):
            world.zero_grad(set_to_none=True)
            rollout_started=time.time();wm_frozen={k:v.clone() for k,v in world.state_dict().items()}
            episode_samples=[];ref_samples=[];phase_samples=[];observable=[];label_samples=[];actions=[]
            reward_sum=0.;true_term=0;timeouts=0;timeout_maxerror=0.
            for step in range(24):
                with torch.no_grad():
                    observable.append(obs[:,:92].clone());label_samples.append(labels().clone())
                    episode_samples.append(episode.clone());ref_samples.append(raw.motion_ids.clone())
                    phase_samples.append((raw.episode_length_buf*raw.step_dt+raw.start_motion_times).clone())
                    value=feature_observation(a.arm,norm,obs,history.value,world)
                    action=algorithm.act(value,value)
                    oldvalue=algorithm.transition.values.clone()
                    nextobs,reward,terminated,truncated,infos=env.step(action)
                    done=terminated|truncated;infos['time_outs']=truncated
                    assert torch.isfinite(reward).all() and torch.isfinite(nextobs['policy']).all()
                    actions.append(action.clone());reward_sum+=float(reward.mean())
                    algorithm.process_env_step(reward,done,infos)
                    expected=reward+.99*oldvalue[:,0]*truncated
                    err=float((algorithm.storage.rewards[step,:,0]-expected).abs().max())
                    timeout_maxerror=max(timeout_maxerror,err);assert err<1e-6
                    true_term+=int(terminated.sum());timeouts+=int(truncated.sum());total_resets+=int(done.sum())
                    episode+=done.int();obs=nextobs['policy'];history.append(obs,done)
            with torch.no_grad():
                observable.append(obs[:,:92].clone());label_samples.append(labels().clone())
                episode_samples.append(episode.clone());ref_samples.append(raw.motion_ids.clone())
                phase_samples.append((raw.episode_length_buf*raw.step_dt+raw.start_motion_times).clone())
                last=feature_observation(a.arm,norm,obs,history.value,world)
                algorithm.compute_returns(last)
                # Epoch0 numerator reads the exact stored histories/latents.
                probe=algorithm.storage.observations[0];policy.act(probe)
                ratio=torch.exp(policy.get_actions_log_prob(algorithm.storage.actions[0])-
                                algorithm.storage.actions_log_prob[0,:,0])
                ratio_error=float((ratio-1).abs().max());assert ratio_error<1e-4
            losses=algorithm.update()
            if a.mode in ['capacity','formal']:
                gradients=[p.grad for p in policy.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                assert all(p.grad is None for p in world.parameters())
            
            assert all(np.isfinite(float(v)) for v in losses.values())
            assert all(torch.equal(v,world.state_dict()[k]) for k,v in wm_frozen.items())
            assert all(torch.equal(v,norm.state_dict()[k]) for k,v in norm_before.items())
            world_after_PPO_hash=tensor_hash(world.state_dict())
            world_before_PPO_hash=tensor_hash(wm_frozen)
            assert world_after_PPO_hash==world_before_PPO_hash
            if a.mode=='formal' and (update==0 or (update+1)%50==0):
                torch.save({'world_before_and_after_PPO':wm_frozen,
                    'world_hash':world_before_PPO_hash,'world_version':world_version,
                    'world_gradients_during_PPO':[p.grad for p in world.parameters()]},
                    a.output/f'world_PPO_boundary_update{update+1:04d}.pt')
            auxaudit=[];handoff_kl=0.;matched=0;eligible=0
            if a.mode=='smoke':
                np.savez_compressed(a.output/f'actual_rollout_update{update:04d}.npz',
                    observable=torch.stack(observable).cpu().numpy(),actions=torch.stack(actions).cpu().numpy(),
                    labels=torch.stack(label_samples).cpu().numpy(),episode=torch.stack(episode_samples).cpu().numpy(),
                    reference=torch.stack(ref_samples).cpu().numpy(),phase=torch.stack(phase_samples).cpu().numpy(),
                    stored_observations=algorithm.storage.observations.cpu().numpy(),
                    stored_actions=algorithm.storage.actions.cpu().numpy(),stored_old_logp=algorithm.storage.actions_log_prob.cpu().numpy(),
                    stored_rewards=algorithm.storage.rewards.cpu().numpy(),stored_dones=algorithm.storage.dones.cpu().numpy())
            online=a.arm.startswith('W-') and a.arm!='W-pretrained-frozen'
            if online:
                bundle,windowaudit=make_windows(torch.stack(observable),torch.stack(actions),torch.stack(label_samples),
                    torch.stack(episode_samples),torch.stack(ref_samples),torch.stack(phase_samples),update*24)
                matched=windowaudit['matched_count'];eligible=windowaudit['eligible_count']
                assert matched>=16,'Insufficient stratified auxiliary windows; preserve failed run'
                if a.mode=='smoke' or update%50==0:
                    np.savez_compressed(a.output/f'aux_window_audit_update{update:04d}.npz',**windowaudit)
                # Fixed audit histories selected without evaluating rewards.
                idxaudit=torch.arange(min(1024,matched),device=a.device)
                historyaudit=bundle['history'][idxaudit]
                # Actor input prefix is the actual observation for each window.
                samplet=torch.as_tensor(windowaudit['sample_t'][:len(idxaudit)]-update*24,device=a.device)
                samplee=torch.as_tensor(windowaudit['sample_env'][:len(idxaudit)],device=a.device)
                baseaudit=algorithm.storage.observations[samplet,samplee,:626].detach().clone()
                policy_frozen={k:v.clone() for k,v in policy.state_dict().items()}
                with torch.no_grad():
                    oldmean=policy.actor(torch.cat([baseaudit,world.encode(historyaudit)],-1)).clone();std=policy.std.clone()
                rng=np.random.default_rng(a.seed+update*1009)
                for _ in range(a.aux_updates):
                    idx=torch.as_tensor(rng.integers(0,matched,size=a.aux_batch),device=a.device)
                    auxaudit.append(auxiliary_step(world,auxiliary,bundle,idx,a.arm in ['W-shuffled-original','W-shuffled-initmatched']))
                    aux_count+=1
                assert all(torch.equal(v,policy.state_dict()[k]) for k,v in policy_frozen.items())
                assert all(torch.equal(v,world.state_dict()[k]) for k,v in ridge_norm.items())
                with torch.no_grad():
                    newmean=policy.actor(torch.cat([baseaudit,world.encode(historyaudit)],-1))
                    handoff_kl=float(gaussian_kl(oldmean,newmean,std))
                assert np.isfinite(handoff_kl) and handoff_kl<=.01,'Encoder handoff KL exceeded locked .01; fail, do not tune/retry'
                world_version+=1
            row={'completed_update':update+1,'actual_transitions':(update+1)*24*a.num_envs,
                'formal_PPO_updates':update+1 if a.mode=='formal' else 0,
                'diagnostic_PPO_updates':update+1 if a.mode!='formal' else 0,
                'training_any_reset_success_fraction_cumulative':float((raw.n_successes>0).float().mean()),
                'training_success_events_cumulative':int(raw.n_successes.sum()),
                'world_before_PPO_hash':world_before_PPO_hash,'world_after_PPO_hash':world_after_PPO_hash,
                'mean_actual_reward':reward_sum/24,'true_terminations':true_term,'timeouts':timeouts,
                'timeout_bootstrap_maxerror':timeout_maxerror,'epoch0_ratio_maxerror':ratio_error,
                'PPO_loss':{k:float(v) for k,v in losses.items()},'learning_rate':algorithm.learning_rate,
                'world_version_before_PPO':world_version-int(online),'world_version_after_aux':world_version,
                'world_unchanged_through_PPO':True,'normalizer_unchanged':True,'aux_updates':aux_count,
                'aux':auxaudit,'handoff_mean_KL':handoff_kl,'eligible_aux_windows':eligible,
                'matched_aux_windows':matched,'control_rows_excluded':0,
                'update_wall_s':time.time()-rollout_started,'total_wall_s':time.time()-started,
                'cuda_peak_MiB':torch.cuda.max_memory_allocated()/1024**2}
            audit.append(row);atomic(a.output/'progress.json',row)
            with (a.output/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            writer.add_scalar('Train/actual_reward_mean',row['mean_actual_reward'],update+1)
            for key,val in row['PPO_loss'].items():writer.add_scalar('Loss/'+key,val,update+1)
            writer.add_scalar('World/encoder_handoff_KL',handoff_kl,update+1)
            writer.add_scalar('Train/cumulative_any_reset_success_fraction',row['training_any_reset_success_fraction_cumulative'],update+1)
            for idx,v in enumerate(auxaudit):writer.add_scalar('World/aux_prediction_loss',v['prediction_loss'],update+1)
            writer.add_scalar('Train/learning_rate',algorithm.learning_rate,update+1);writer.flush()
            print(f'WORLD_PPO_UPDATE {update+1}/{a.updates}',json.dumps(row),flush=True)
            if (update+1)%50==0 or update+1==a.updates:save_checkpoint(update+1)
        summary={'status':'COMPLETE','mode':a.mode,'arm':a.arm,'seed':a.seed,'num_envs':a.num_envs,
            'updates':a.updates,'actual_transitions':a.num_envs*24*a.updates,
            'formal_PPO_updates':a.updates if a.mode=='formal' else 0,
            'diagnostic_PPO_updates':a.updates if a.mode!='formal' else 0,'aux_updates':aux_count,
            'restored_optimizer':restored,'initial_mean_maxerror':error_mean,'initial_value_maxerror':error_value,
            'initial_std_maxerror':error_std,'initial_logprob_maxerror':error_logp,'reference_maxerror':ref_error,
            'original_checkpoint_sha256':EXPECTED,'selected_world_checkpoint_sha256':sha(fit/('residual_shuffled.pt' if a.arm=='W-shuffled-original' else 'residual_correct.pt')),
            'initial_world_hash':world_initial,'final_world_hash':tensor_hash(world.state_dict()),
            'policy_changed':policy_before!=tensor_hash(policy.state_dict()),'norm_frozen':True,
            'reset_count':total_resets,'no_extra_IK':True,'imagined_transitions':0,
            'original_task_reward_PD_termination_unchanged':True,
            'timeout_semantics':'original single pre-action value bootstrap; not exact terminal-state value',
            'schema_hash':SCHEMA_HASH,'world_model_prediction_pass_is_not_control_gate':True,
            'parameter_counts':{'PPO_trainable':sum(p.numel() for p in policy.parameters() if p.requires_grad),
                'aux_trainable':sum(p.numel() for p in world.parameters()) if a.arm.startswith('W-') else 0,
                'deployed_actor':sum(p.numel() for p in policy.actor.parameters())+policy.std.numel()+
                 (sum(p.numel() for p in policy.history_encoder.parameters()) if a.arm in ['P-cap','P-history'] else
                  sum(p.numel() for p in world.encoder.parameters()) if a.arm.startswith('W-') else 0)},
            'innovation_validated':False,'wall_s':time.time()-started,'audit':audit}
        assert summary['policy_changed']
        atomic(a.output/'summary.json',summary)
        print('WORLD_RESIDUAL_PPO_CONTROL_COMPLETE',flush=True)
    finally:
        writer.close();env.close();faulthandler.cancel_dump_traceback_later()

if __name__=='__main__':
    try:main()
    finally:app.close()
