"""Read-only Pick policy evaluation; separate first-episode success from training completion."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103'
sys.path.insert(0,str(S/'src'))
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--num-envs',type=int,default=100)
parser.add_argument('--steps',type=int,default=500)
parser.add_argument('--smoke',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.output_dir.mkdir(exist_ok=False)
app=None;env=None
try:
    import hashlib
    checkpoint_sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    active=json.loads((S/'active_pick_run.json').read_text()) if (S/'active_pick_run.json').exists() else None
    run=Path(active['run_dir']) if active else S/'runs/pick29_8912_5000_v1'
    target=int(active['total_updates']) if active else 5000
    training=json.loads((run/'status.json').read_text())
    if not args.smoke:
        assert training['status']=='ppo_completed' and training['policy_updates']==target
        assert args.checkpoint.resolve()==(run/'ppo'/('model_%d.pt'%(target-1))).resolve()
        assert args.num_envs==100 and args.steps>=500
    began=time.monotonic();app=AppLauncher(args).app
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    import stage3_pick29_env as base
    from stage3_pick29_core import sha
    torch.cuda.set_per_process_memory_fraction(.05,device=args.device)
    rng=torch.Generator(device=args.device).manual_seed(20261104)

    def randomized_reset(e,env_ids):
        if env_ids is None:env_ids=torch.arange(e.num_envs,device=e.device)
        base.reset_scene(e,env_ids)
        obj=e.scene['object'];root=obj.data.default_root_state[env_ids].clone()
        root[:,:3]+=e.scene.env_origins[env_ids]
        # Project-baseline range fits the calibrated table. It is explicit, not
        # claimed to equal the paper's full test-position distribution.
        root[:,:2]+=(torch.rand((len(env_ids),2),device=e.device,generator=rng)*2-1)*.06
        root[:,7:]=0
        obj.write_root_state_to_sim(root,env_ids=env_ids)

    def capture_success(e):
        value=base.pick_success(e)
        e._evaluation_success=value.bool().clone()
        e._evaluation_height=(e.scene['object'].data.root_pos_w[:,2]-e.scene.env_origins[:,2]).clone()
        body=e.scene['robot'].find_bodies('right_wrist_yaw_link')[0][0]
        e._evaluation_wrist=(e.scene['robot'].data.body_pos_w[:,body]-e.scene.env_origins).clone()
        return value

    cfg=base.Pick29EnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=20261104
    cfg.reference_manifest=str(S/'reports/native_table_replay_final/reference_bank.json')
    cfg.events.reset=EventTermCfg(func=randomized_reset,mode='reset')
    cfg.rewards.pick.func=capture_success
    env=ManagerBasedRLEnv(cfg=cfg)
    wrapped=RslRlVecEnvWrapper(env)
    rcfg=G1FlatPPORunnerCfg()
    runner=OnPolicyRunner(wrapped,rcfg.to_dict(),log_dir=None,device=args.device)
    runner.load(str(args.checkpoint),load_optimizer=False)
    actor=runner.get_inference_policy(device=args.device)
    obs,_=wrapped.reset()
    assert obs.shape==(args.num_envs,102)
    initial=env.scene['object'].data.root_pos_w.clone()-env.scene.env_origins
    goal=initial+torch.tensor([0.,0.,.04],device=env.device)
    grasp_times=env._reference_bank.tg[env._reference_ids].clone()
    active=torch.ones(args.num_envs,device=env.device,dtype=torch.bool)
    success=torch.zeros_like(active);returns=torch.zeros(args.num_envs,device=env.device)
    duration=torch.zeros(args.num_envs,device=env.device,dtype=torch.long)
    contact_error=torch.full((args.num_envs,),float('nan'),device=env.device)
    max_height=initial[:,2].clone()
    for step in range(args.steps):
        before=env.episode_length_buf*env.step_dt
        with torch.inference_mode():
            action=actor(obs)
            assert action.shape==(args.num_envs,31) and torch.isfinite(action).all()
            obs,reward,done,extras=wrapped.step(action)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        success|=active & env._evaluation_success
        returns+=reward*active
        duration+=active.long()
        max_height=torch.where(active,torch.maximum(max_height,env._evaluation_height),max_height)
        crossing=active & (before<grasp_times) & (before+env.step_dt>=grasp_times)
        error=(env._evaluation_wrist-goal).norm(dim=-1)
        contact_error=torch.where(crossing,error,contact_error)
        active&=~done.bool()
        if not bool(active.any()):break
    finite=torch.isfinite(contact_error)
    result={'status':'evaluation_completed' if not args.smoke else 'entrypoint_smoke_completed',
        'final_evaluation':not args.smoke,'num_envs':args.num_envs,'steps_executed':step+1,
        'planned_training_updates':target,
        'checkpoint':str(args.checkpoint),'checkpoint_sha256':checkpoint_sha,
        'policy_updates_during_evaluation':0,'body_dof':29,'actor_obs':102,'policy_actions':31,
        'actor_reference_input':False,'success_count':int(success.sum()),
        'success_ratio':float(success.float().mean()),'first_episodes_finished':int((~active).sum()),
        'mean_first_episode_return':float(returns.mean()),'mean_episode_steps':float(duration.float().mean()),
        'mean_contact_error_m_among_episodes_reaching_tg':float(contact_error[finite].mean()) if bool(finite.any()) else None,
        'episodes_reaching_tg':int(finite.sum()),'episodes_not_reaching_tg':int((~finite).sum()),
        'object_initial_positions':initial.cpu().tolist(),'max_object_height_m':max_height.cpu().tolist(),
        'object_xy_uniform_half_width_m':.06,'random_seed':20261104,'elapsed_s':time.monotonic()-began,
        'reference_bank_sha256':env._reference_bank.manifest_sha256,'asset_sha256':sha(base.ASSET),
        'scope':str(args.num_envs)+' randomized project-baseline cases; actual height>0.95 after grasp time; not an exact paper comparison or physical deployment',
        'training_budget_complete_is_not_task_success':True}
    assert sha(args.checkpoint)==checkpoint_sha,'evaluation modified checkpoint'
    with (args.output_dir/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('PICK_EVALUATION '+json.dumps(result),flush=True)
except BaseException as e:
    (args.output_dir/'failure.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()},indent=2))
    raise
finally:
    if env is not None:env.close()
    if app is not None:app.close(wait_for_replicator=False)
