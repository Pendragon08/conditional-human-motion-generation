"""Real Isaac data collection and a bounded WM/PPO interface witness.

Warm-up never updates PPO. Smoke really updates PPO but is diagnostic only.
Unmodified task reward, PD, termination and references; no additional IK.
"""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['smoke','collect','visibility'], required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--num_envs', type=int, default=8192)
p.add_argument('--rollouts', type=int, default=20)
p.add_argument('--visibility-seconds', type=int, default=1200)
p.add_argument('--disturbance', choices=['none','lateral80'], default='none')
p.add_argument('--monitor-prefix', type=Path)
p.add_argument('--reference-manifest', type=Path)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
sys.argv = [sys.argv[0]]
launcher = AppLauncher(a)
app = launcher.app

import hashlib
import json
import time
import traceback
import faulthandler
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder
from isaaclab.utils.math import quat_apply_inverse
from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from rsl_rl.algorithms import PPO
from r3_i1_training_telemetry import SampledTrainingTelemetry
from r3_world_ppo_core import WorldModel, HistoryBuffer, AugmentedActorCritic, prediction_loss
from r3_world_ppo_core import INPUT, HISTORY, HORIZONS, SCHEMA, SCHEMA_HASH

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt')
JIT = CKPT.parent/'exported/policy.pt'
EXPECTED = 'e3c50fb6f36f59f89d1b4e788b1ba74cf1ff0cdde3dceff34a9d7765235f65c0'

def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(f, data):
    tmp=f.with_suffix(f.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(f)

def main():
    if a.output.exists(): raise RuntimeError('Output exists: preserve, do not duplicate')
    a.output.mkdir(parents=True)
    started=time.time()
    faulthandler.enable()
    faulthandler.dump_traceback_later(180,repeat=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    assert sha(CKPT)==EXPECTED
    checkpoint=torch.load(CKPT,map_location=a.device,weights_only=False)
    original=ActorCritic(626,626,29,actor_hidden_dims=[512,256,256],critic_hidden_dims=[512,256,256],activation='elu').to(a.device)
    original.load_state_dict(checkpoint['model_state_dict'])
    original.eval()
    norm=EmpiricalNormalization(626).to(a.device)
    norm.load_state_dict(checkpoint['obs_norm_state_dict']); norm.eval()
    privnorm=EmpiricalNormalization(626).to(a.device)
    privnorm.load_state_dict(checkpoint['privileged_obs_norm_state_dict']); privnorm.eval()
    cfg=parse_env_cfg('Isaac-Motion-Tracking-Kick-v0',device=a.device,num_envs=a.num_envs)
    cfg.seed=a.seed
    env=gym.make('Isaac-Motion-Tracking-Kick-v0',cfg=cfg)
    env=SampledTrainingTelemetry(env,'127.0.0.1:40123')
    raw=env.unwrapped; robot=raw.scene['robot']; lib=raw.motion_lib
    assert len(JointNamesOrder)==27 and raw.step_dt==.02
    jid=[robot.joint_names.index(x) for x in JointNamesOrder]
    feet=[robot.body_names.index(x) for x in ['left_ankle_roll_link','right_ankle_roll_link']]
    sensor=raw.scene['contact_forces']
    cid=[sensor.body_names.index(x) for x in ['left_ankle_roll_link','right_ankle_roll_link']]
    obs,_=env.reset(seed=a.seed); obs=obs['policy']
    assert obs.shape==(a.num_envs,626)
    # Original task has no privileged critic observation; official runner then
    # evaluates its critic on actor-normalized observations. The unused critic
    # normalizer in the checkpoint has count=0 and must not normalize raw obs.
    assert 'critic' not in raw.observation_manager.active_terms
    assert int(checkpoint['privileged_obs_norm_state_dict']['count'])==0
    privnorm=norm
    reference_receipt=None
    if a.reference_manifest:
        reference_receipt=json.loads(a.reference_manifest.read_text())
        assert reference_receipt['status']=='COMPLETE' and reference_receipt['selected_count']==11
        loaded_paths=[Path(f) for f in lib._motion_data_load]
        assert {f.name for f in loaded_paths}=={r['file'] for r in reference_receipt['selected']}
        for row in reference_receipt['selected']:
            f=Path(reference_receipt['reference_dir'])/row['file']
            assert sha(f)==row['sha256'] and f.stat().st_size==row['bytes']
        expected_q=np.concatenate([np.asarray(d['joints'])[:200] for d in lib._motion_data_list])
        identity=float(np.max(np.abs(lib.dof_pos.cpu().numpy()-expected_q)))
    else:
        official=np.load(ROOT/'artifacts/sim2sim_mujoco/interfaces/Kick_references.npz')
        identity=float(np.max(np.abs(lib.dof_pos.cpu().numpy()-official['dof_pos'])))
    assert identity<1e-6
    with torch.no_grad():
        jit=torch.jit.load(str(JIT),map_location=a.device).eval()
        initial_error=float((original.act_inference(norm(obs))-jit(obs)).abs().max())
        assert initial_error<1e-5
    active=ROOT/'artifacts/monitoring/r3_active_dreamcontrol_training.json'
    run_id=a.output.name[len(f'{a.mode}_seed{a.seed}_'):]
    prefix=a.monitor_prefix or ROOT/f'artifacts/monitoring/r3_world_ppo_{a.mode}_{a.seed}_{run_id}'
    atomic_json(active,{'task_id':'Isaac-Motion-Tracking-Kick-v0', 'stage':a.mode,
                        'is_formal_training':False,'source':'actual_isaaclab_simulation',
                        'output':str(a.output),'seed':a.seed,'num_envs':a.num_envs,
                        'queue_state':str(prefix.with_suffix('.state')),
                        'log':str(prefix.with_suffix('.log'))})
    history=HistoryBuffer(a.num_envs,a.device)
    history.append(obs)
    wm=WorldModel().to(a.device)
    smoke_algorithm=None
    versions=[]
    if a.mode=='smoke':
        # Five-arm initial functions, architecture and normalization checks.
        init_functions={}
        init_encoder={k:v.detach().clone() for k,v in wm.encoder.state_dict().items()}
        with torch.no_grad():
            baseobs=norm(obs); latent=wm.encode(history.value)
            ref_action=original.act(baseobs)
            ref_mean=original.action_mean.clone();ref_logp=original.get_actions_log_prob(ref_action)
            ref_value=original.evaluate(baseobs)
            for arm in ['P0','P-cap','P-history','W-correct','W-shuffled']:
                trial=AugmentedActorCritic(arm,checkpoint['model_state_dict'],init_encoder).to(a.device)
                if arm=='P0':value=baseobs
                elif arm=='P-history':value=torch.cat([baseobs,history.value.flatten(1)],-1)
                elif arm=='P-cap':value=torch.cat([baseobs,obs[:,:INPUT]],-1)
                else:value=torch.cat([baseobs,latent],-1)
                trial.act(value)
                errors={'action_mean':float((trial.action_mean-ref_mean).abs().max()),
                        'value':float((trial.evaluate(value)-ref_value).abs().max()),
                        'log_probability':float((trial.get_actions_log_prob(ref_action)-ref_logp).abs().max())}
                assert errors['action_mean']<1e-5 and errors['value']<1e-5 and errors['log_probability']<1e-4
                init_functions[arm]=errors
        fixture=HistoryBuffer(a.num_envs,a.device)
        fixture.append(obs);fixture.append(obs)
        mask=torch.zeros(a.num_envs,dtype=torch.bool,device=a.device);mask[0]=True
        fixture.append(obs,mask)
        assert torch.count_nonzero(fixture.value[0,:-1])==0 and torch.equal(fixture.value[0,-1],obs[0,:INPUT])
        augmented=AugmentedActorCritic('W-correct',checkpoint['model_state_dict']).to(a.device)
        with torch.no_grad():
            latent=wm.encode(history.value)
            initial_aug_error=float((augmented.act_inference(torch.cat([norm(obs),latent],-1))-
                                     original.act_inference(norm(obs))).abs().max())
            print('WORLD_PPO_INITIAL_LATENT_FUNCTION_ERROR',initial_aug_error,flush=True)
            # A wider GEMM can change accumulation order even with zero columns.
            assert initial_aug_error<1e-5
        smoke_algorithm=PPO(augmented,device=a.device,num_learning_epochs=5,num_mini_batches=4,
                            clip_param=.2,gamma=.99,lam=.95,value_loss_coef=1.,entropy_coef=.008,
                            learning_rate=.001,max_grad_norm=1.,use_clipped_value_loss=True,
                            schedule='adaptive',desired_kl=.01)
        smoke_algorithm.init_storage('rl',a.num_envs,24,[754],[754],[29])
        # Isolated timeout semantic fixture, not a changed simulation rollout.
        with torch.no_grad():
            probe=torch.cat([norm(obs),wm.encode(history.value)],-1)
            smoke_algorithm.act(probe,probe)
            oldvalue=smoke_algorithm.transition.values.clone()
            timeout=torch.zeros(a.num_envs,device=a.device);timeout[0]=1
            done_fixture=timeout.bool()
            smoke_algorithm.process_env_step(torch.zeros(a.num_envs,device=a.device),done_fixture,{'time_outs':timeout})
            expected=(.99*oldvalue[:,0]*timeout)
            assert torch.allclose(smoke_algorithm.storage.rewards[0,:,0],expected,atol=1e-6,rtol=1e-6)
            smoke_algorithm.storage.clear()
        prior_params={k:v.detach().clone() for k,v in augmented.state_dict().items()}
    steps=a.rollouts*24
    if a.mode=='visibility': steps=10**8
    arrays={}
    if a.mode!='visibility':
        specs={'observable':((steps+1,a.num_envs,INPUT),np.float32),
               'labels':((steps+1,a.num_envs,50),np.float32),
               'actions':((steps,a.num_envs,29),np.float32),
               'done':((steps,a.num_envs),np.bool_),
               'reference':((steps+1,a.num_envs),np.int16),
               'episode':((steps+1,a.num_envs),np.int32),
               'phase':((steps+1,a.num_envs),np.float32),
               'reward':((steps,a.num_envs),np.float32)}
        arrays={k:np.lib.format.open_memmap(a.output/(k+'.npy'),mode='w+',dtype=d,shape=s)
                for k,(s,d) in specs.items()}
    episode=torch.zeros(a.num_envs,dtype=torch.int32,device=a.device)
    losses=[]; resets=0; wm_version=0; last_done=None
    # New, explicitly separate disturbance development data. No force state is
    # included in the policy or encoder. Even envs are clean controls; odd envs
    # receive one signed world-y torso force at controls 50..59 inclusive.
    order=np.lexsort((np.arange(a.num_envs),
        np.floor((raw.episode_length_buf*raw.step_dt+raw.start_motion_times).cpu().numpy()/.2),
        raw.motion_ids.cpu().numpy()))
    ranks=torch.tensor(np.argsort(order),device=a.device)
    cohort=ranks%2
    direction=torch.where((ranks//2)%2==0,1.,-1.)
    torso=robot.body_names.index('torso_link')
    push_episode=None; reset_during_push=0; reset_after_push=0; force_audit=[]
    def labels():
        rel=robot.data.body_pos_w[:,feet]-robot.data.root_pos_w[:,None,:]
        qb=robot.data.root_quat_w[:,None,:].expand(-1,2,-1)
        relb=quat_apply_inverse(qb.reshape(-1,4),rel.reshape(-1,3)).reshape(a.num_envs,6)
        contact=(sensor.data.net_forces_w[:,cid].norm(dim=-1)>5).float()
        value=torch.cat([robot.data.joint_pos[:,jid],robot.data.projected_gravity_b,
                         robot.data.root_lin_vel_b,robot.data.root_ang_vel_b,relb,
                         robot.data.body_lin_vel_w[:,feet,:2].flatten(1),
                         (robot.data.root_pos_w[:,2]-raw.scene.env_origins[:,2])[:,None],
                         robot.data.root_vel_w[:,2:3],contact],-1)
        assert value.shape[-1]==50 and torch.isfinite(value).all()
        return value
    def store_state(t):
        arrays['observable'][t]=obs[:,:INPUT].detach().cpu().numpy()
        arrays['labels'][t]=labels().detach().cpu().numpy()
        arrays['reference'][t]=raw.motion_ids.cpu().numpy()
        arrays['episode'][t]=episode.cpu().numpy()
        arrays['phase'][t]=(raw.episode_length_buf*raw.step_dt+raw.start_motion_times).cpu().numpy()
    if arrays: store_state(0)
    for step in range(steps):
        if a.disturbance=='lateral80':
            pulse_step=step%500 if a.mode=='visibility' else step
            world=torch.zeros(a.num_envs,3,device=a.device)
            if 50<=pulse_step<60:
                world[:,1]=80.*cohort*direction
                if step==50: push_episode=episode.detach().cpu().numpy().copy()
            local=quat_apply_inverse(robot.data.body_quat_w[:,torso],world)[:,None,:]
            robot.set_external_force_and_torque(local,torch.zeros_like(local),body_ids=[torso])
            if 50<=step<60:
                applied=robot._external_force_b[:,torso]
                error=float((applied-local[:,0]).abs().max())
                assert error<1e-6
                force_audit.append({'step':step,'local_buffer_error':error,
                    'pushed_count':int((applied.norm(dim=-1)>1).sum()),
                    'force_norm_N_max':float(applied.norm(dim=-1).max())})
        with torch.inference_mode():
            if smoke_algorithm:
                latent=wm.encode(history.value)
                augmented_obs=torch.cat([norm(obs),latent],-1)
                augmented_critic=torch.cat([privnorm(obs),latent],-1)
                action=smoke_algorithm.act(augmented_obs,augmented_critic)
            else:
                # Stochastic frozen original policy: exploration std is the
                # original checkpoint std; no new PPO/warm-up learning.
                action=original.act(norm(obs))
        result=env.step(action)
        nextobs,reward,terminated,truncated,infos=result
        done=terminated|truncated
        if a.disturbance=='lateral80':
            if 50<=step<60: reset_during_push+=int(done[cohort.bool()].sum())
            elif 60<=step<110: reset_after_push+=int(done[cohort.bool()].sum())
        assert torch.isfinite(nextobs['policy']).all() and torch.isfinite(reward).all()
        if arrays:
            arrays['actions'][step]=action.detach().cpu().numpy()
            arrays['reward'][step]=reward.detach().cpu().numpy()
            arrays['done'][step]=done.cpu().numpy()
        if smoke_algorithm:
            infos['time_outs']=truncated
            smoke_algorithm.process_env_step(reward,done,infos)
        episode+=done.int(); resets+=int(done.sum())
        obs=nextobs['policy']
        history.append(obs,done)
        if arrays: store_state(step+1)
        if smoke_algorithm and (step+1)%24==0:
            frozen={k:v.detach().clone() for k,v in wm.state_dict().items()}
            with torch.inference_mode():
                last_latent=wm.encode(history.value)
                smoke_algorithm.compute_returns(torch.cat([privnorm(obs),last_latent],-1))
            out=smoke_algorithm.update()
            assert all(torch.equal(v,wm.state_dict()[k]) for k,v in frozen.items())
            assert all(np.isfinite(float(v)) for v in out.values())
            losses.append({k:float(v) for k,v in out.items()})
            versions.append({'rollout':step//24,'wm_version':wm_version,
                             'actor_norm_version':'original_frozen',
                             'wm_norm_version':0,'schema_hash':SCHEMA_HASH,
                             'world_model_unchanged_through_ppo_epochs':True})
        if step%24==0:
            atomic_json(a.output/'progress.json',{'stage':a.mode,'steps':step+1,
                        'wall_s':time.time()-started,'actual_transitions':(step+1)*a.num_envs,
                        'formal_ppo_updates':0,'diagnostic_ppo_updates':len(losses)})
            print('WORLD_PPO_ACTUAL_PHYSICS_STEP',step+1,flush=True)
        if a.mode=='visibility':
            time.sleep(.02)
            if time.time()-started>a.visibility_seconds: break
    for v in arrays.values(): v.flush()
    if a.disturbance=='lateral80' and arrays:
        assert push_episode is not None
        np.save(a.output/'cohort.npy',cohort.cpu().numpy().astype(np.int8))
        np.save(a.output/'push_direction.npy',direction.cpu().numpy().astype(np.int8))
        np.save(a.output/'push_episode.npy',push_episode)
    if smoke_algorithm:
        # One CUDA auxiliary gradient update using genuine, reset-free data.
        valid=[]
        for t in range(HISTORY-1,steps-10):
            for e in range(a.num_envs):
                if not arrays['done'][t-HISTORY+1:t+10,e].any(): valid.append((t,e))
        assert valid
        pairs=np.asarray(valid[:256]); tt,ee=pairs.T
        xx=torch.tensor(np.stack([arrays['observable'][t-HISTORY+1:t+1,e] for t,e in pairs]),device=a.device)
        aa=torch.tensor(np.stack([arrays['actions'][t:t+10,e] for t,e in pairs]),device=a.device)
        yy={k:torch.tensor(np.stack([np.concatenate([
            arrays['labels'][t+k,e,:48]-arrays['labels'][t,e,:48],
            arrays['labels'][t+k,e,48:]]) for t,e in pairs]),device=a.device) for k in HORIZONS}
        opt=torch.optim.Adam(wm.parameters(),lr=.001)
        before={k:v.detach().clone() for k,v in wm.named_parameters()}
        loss,_=prediction_loss(wm,xx,aa,yy); opt.zero_grad(); loss.backward(); opt.step()
        wm_changed=any(not torch.equal(v,dict(wm.named_parameters())[k]) for k,v in before.items())
        assert wm_changed and torch.isfinite(loss)
        policy_changed=any(not torch.equal(v,augmented.state_dict()[k]) for k,v in prior_params.items())
        assert policy_changed
        torch.save({'actor_critic':augmented.state_dict(),'wm':wm.state_dict(),
                    'base_obs_norm':norm.state_dict(),'feature_schema':SCHEMA,
                    'schema_hash':SCHEMA_HASH,'diagnostic_only':True},a.output/'diagnostic_checkpoint.pt')
    summary={'status':'COMPLETE','mode':a.mode,'platform':'Isaac Lab / PhysX',
             'seed':a.seed,'num_envs':a.num_envs,'steps':step+1,
             'actual_transitions':(step+1)*a.num_envs,'formal_ppo_updates':0,
             'diagnostic_ppo_updates':len(losses),'diagnostic_aux_updates':int(bool(smoke_algorithm)),
             'task_reward_PD_termination_unchanged':True,'extra_IK':False,
             'shared_warmup_stochastic_original_policy':a.mode=='collect',
             'policy_checkpoint_sha256':EXPECTED,'original_jit_equivalence_max_abs':initial_error,
             'original_library_order_max_abs':identity,'reset_count':resets,
             'schema':SCHEMA,'schema_hash':SCHEMA_HASH,'rollout_versions':versions,
             'losses':losses,'wall_s':time.time()-started}
    if reference_receipt:
        summary.update(reference_manifest_sha256=sha(a.reference_manifest),
                       fresh_confirmation_references=True,
                       loaded_reference_files=[f.name for f in loaded_paths],
                       selected_original_ids=[r['original_id'] for r in reference_receipt['selected']],
                       confirmation_data_not_for_training=True)
    summary['disturbance_protocol']={'name':a.disturbance,
        'distribution_changed_from_nominal':a.disturbance!='none',
        'force_world_y_N':80 if a.disturbance=='lateral80' else 0,
        'force_control_steps':[50,59] if a.disturbance=='lateral80' else [],
        'force_duration_s':.2 if a.disturbance=='lateral80' else 0.,
        'cohort':'rank original reference/initial phase; alternating clean/pushed and force signs',
        'external_force_visible_to_actor':False,
        'actual_articulation_force_buffer_audit':force_audit,
        'reset_during_force_pushed_envs':reset_during_push,
        'reset_60_to_109_pushed_envs':reset_after_push}
    if smoke_algorithm:
        summary.update({'zero_initialized_latent_initial_actor_error':initial_aug_error,
                        'five_arm_initial_function_errors':init_functions,
                        'synthetic_history_reset_fixture_passed':True,
                        'official_single_timeout_bootstrap_fixture_passed':True,
                        'critic_uses_actor_norm_like_original_nonprivileged_runner':True,
                        'policy_weights_really_updated':policy_changed,
                        'world_weights_really_updated':wm_changed,
                        'aux_loss':float(loss.detach())})
    summary['files']={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)}
                      for f in a.output.glob('*.npy')}
    atomic_json(a.output/'summary.json',summary)
    print('WORLD_PPO_ISAAC_COMPLETE',json.dumps(summary),flush=True)
    env.close()

try:
    main()
except BaseException:
    traceback.print_exc()  # preserve exception before potentially slow Kit close
    raise
finally:
    app.close()
