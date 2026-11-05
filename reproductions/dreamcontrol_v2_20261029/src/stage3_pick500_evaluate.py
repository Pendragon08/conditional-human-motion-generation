"""Read-only evaluation of the new500-reference Pick PPO, not the old baseline.

Training pairs object/reset poses to references. Autonomous evaluation may sample
fresh object positions: reference is then only an initializer/reward comparator,
never an actor input. Contact error is measured against the actual object goal.
"""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103'
sys.path.insert(0,str(S/'src'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--training-run',type=Path,required=True)
p.add_argument('--reference-manifest',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=100)
p.add_argument('--steps',type=int,default=500)
p.add_argument('--object-reset-mode',choices=['fresh_uniform_positions','reference_positions'],default='fresh_uniform_positions')
p.add_argument('--integration-test',action='store_true')
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output_dir.mkdir(exist_ok=False)
app=env=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
    checkpoint_sha=sha(a.checkpoint);manifest_sha=sha(a.reference_manifest)
    training=json.loads((a.training_run/'status.json').read_text())
    if a.integration_test:assert a.num_envs==4 and a.steps<=32 and training['policy_updates']==2
    else:
        assert training['status']=='ppo_completed' and training['policy_updates']==3000
        assert a.checkpoint.resolve()==(a.training_run/'ppo/model_2999.pt').resolve()
        assert a.num_envs==100 and a.steps>=500 and training['reference_count']==500
    began=time.monotonic();app=AppLauncher(a).app
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    import stage3_pick29_env as base
    import stage3_pick500_env as paired
    from stage3_pick29_core import BODY_JOINTS
    torch.cuda.set_per_process_memory_fraction(.08,device=a.device);torch.set_num_threads(2)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True);provenance=saved['infos']['provenance']
    assert provenance['reference_bank_sha256']==manifest_sha and provenance['asset_sha256']==sha(base.ASSET)
    assert provenance['scene_sha256']==base.SCENE_SHA and provenance['policy_actions']==31 and provenance['policy_observations']==102
    assert provenance['object_pose_policy']=='per_reference_fixed' and not provenance['actor_reference_input']
    rng=torch.Generator(device=a.device).manual_seed(20261105)
    def evaluation_reset(e,env_ids):
        if env_ids is None:env_ids=torch.arange(e.num_envs,device=e.device)
        paired.reset_scene(e,env_ids)
        if a.object_reset_mode=='fresh_uniform_positions':
            obj=e.scene['object'];root=obj.data.default_root_state[env_ids].clone()
            root[:,:3]+=e.scene.env_origins[env_ids]
            root[:,:2]+=(torch.rand((len(env_ids),2),device=e.device,generator=rng)*2-1)*.06
            root[:,7:]=0;obj.write_root_state_to_sim(root,env_ids=env_ids)
    def capture(e):
        value=base.pick_success(e);e._evaluation_success=value.bool().clone()
        e._evaluation_object=(e.scene['object'].data.root_pos_w-e.scene.env_origins).clone()
        idx=e.scene['robot'].find_bodies('right_wrist_yaw_link')[0][0]
        e._evaluation_wrist=(e.scene['robot'].data.body_pos_w[:,idx]-e.scene.env_origins).clone()
        return value
    cfg=paired.Pick500EnvCfg();cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=20261105
    cfg.reference_manifest=str(a.reference_manifest)
    if a.integration_test:cfg.minimum_reference_count=2
    cfg.events.reset=EventTermCfg(func=evaluation_reset,mode='reset');cfg.rewards.pick.func=capture
    env=ManagerBasedRLEnv(cfg=cfg);wrapped=RslRlVecEnvWrapper(env)
    runner=OnPolicyRunner(wrapped,G1FlatPPORunnerCfg().to_dict(),log_dir=None,device=a.device)
    runner.load(str(a.checkpoint),load_optimizer=False);actor=runner.get_inference_policy(device=a.device)
    obs,_=wrapped.reset();assert obs.shape==(a.num_envs,102)
    private_ids=env._reference_ids.clone();live=base.live_observation(env).clone()
    env._reference_ids=(private_ids+1)%env._reference_bank.count
    leak=float((base.live_observation(env)-live).abs().max());env._reference_ids=private_ids;assert leak==0.
    assert env.action_manager.total_action_dim==31 and env.scene['robot'].find_joints(BODY_JOINTS,preserve_order=True)[1]==BODY_JOINTS
    def state_hash():
        h=hashlib.sha256()
        for k,v in sorted(runner.alg.policy.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    policy_before=state_hash();initial=(env.scene['object'].data.root_pos_w-env.scene.env_origins).clone()
    assert bool(((initial[:,:2]-torch.tensor([.35,-.2],device=a.device)).abs()<=.060001).all())
    assert float(initial[:,2].max())<.95
    goal=initial+torch.tensor([0.,0.,.04],device=a.device)
    grasp_steps=torch.ceil(env._reference_bank.tg[private_ids].to(torch.float64)/env.step_dt-1e-4).long()
    active=torch.ones(a.num_envs,device=a.device,dtype=torch.bool);success=torch.zeros_like(active)
    returns=torch.zeros(a.num_envs,device=a.device);duration=torch.zeros(a.num_envs,device=a.device,dtype=torch.long)
    error=torch.full((a.num_envs,),float('nan'),device=a.device);height=initial[:,2].clone()
    sustained=torch.zeros(a.num_envs,device=a.device,dtype=torch.long);longest=torch.zeros_like(sustained)
    for step in range(a.steps):
        before=env.episode_length_buf.clone()
        with torch.inference_mode():
            action=actor(obs);assert action.shape==(a.num_envs,31) and torch.isfinite(action).all()
            obs,reward,done,extras=wrapped.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        success|=active&env._evaluation_success;returns+=active*reward;duration+=active.long()
        height=torch.where(active,torch.maximum(height,env._evaluation_object[:,2]),height)
        crossing=active&(before<grasp_steps)&(before+1>=grasp_steps)
        error=torch.where(crossing,(env._evaluation_wrist-goal).norm(dim=-1),error)
        sustained=torch.where(active&env._evaluation_success,sustained+1,torch.zeros_like(sustained));longest=torch.maximum(longest,sustained)
        active&=~done.bool()
        if not bool(active.any()):break
    assert state_hash()==policy_before and sha(a.checkpoint)==checkpoint_sha
    finite=torch.isfinite(error)
    nearest=torch.cdist(initial[:,:2],env._reference_bank.object_positions[:,:2]).amin(dim=1)
    result={'status':'integration_evaluation_completed' if a.integration_test else 'evaluation_completed','final_evaluation':not a.integration_test,
        'num_envs':a.num_envs,'steps_executed':step+1,'object_reset_mode':a.object_reset_mode,'random_seed':20261105,
        'checkpoint':str(a.checkpoint),'checkpoint_sha256':checkpoint_sha,'reference_bank_sha256':manifest_sha,
        'asset_sha256':sha(base.ASSET),'scene_sha256':base.SCENE_SHA,'reference_count':env._reference_bank.count,
        'body_dof':29,'actor_obs':102,'policy_actions':31,'actor_reference_input':False,'actor_reference_mutation_difference':leak,
        'success_count':int(success.sum()),'success_ratio':float(success.float().mean()),
        'success_definition':'object center height>0.95m after reference grasp time; report as height-based task criterion, not certified finger grasp',
        'first_episodes_finished':int((~active).sum()),'mean_first_episode_return':float(returns.mean()),
        'mean_contact_error_m_among_episodes_reaching_tg':float(error[finite].mean()) if bool(finite.any()) else None,
        'episodes_reaching_tg':int(finite.sum()),'episode_steps':duration.cpu().tolist(),'max_object_height_m':height.cpu().tolist(),
        'longest_consecutive_height_success_seconds':(longest*env.step_dt).cpu().tolist(),
        'object_initial_positions':initial.cpu().tolist(),'nearest_training_object_xy_distance_m':nearest.cpu().tolist(),
        'private_initialization_reference_ids':private_ids.cpu().tolist(),'policy_updates_during_evaluation':0,'policy_state_hash_unchanged':True,
        'evaluation_note':'fresh uniform object targets measure autonomous execution; private reference initializes robot and computes unused evaluation rewards, not control input',
        'scope':'current tabletop uniformXY+-0.06m simulation baseline; not full paper test distribution or hardware deployment',
        'elapsed_seconds':time.monotonic()-began,'training_budget_complete_is_not_task_success':True}
    (a.output_dir/'report.json').open('x').write(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)
except BaseException as e:
    failure={'status':'failed_preserved','error':repr(e),'traceback':traceback.format_exc()}
    (a.output_dir/'failure.json').write_text(json.dumps(failure,indent=2));print(failure['traceback'],flush=True);raise
finally:
    if env is not None:env.close()
    if app is not None:app.close(wait_for_replicator=False)
