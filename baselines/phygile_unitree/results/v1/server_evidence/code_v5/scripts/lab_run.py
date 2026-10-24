"""Explicit Unitree RL Lab/PhyGile integration and bounded GUI continuation.

Preview/inference and PPO continuation are separate modes, with fresh outputs.
Original checkpoints and data are opened read-only. Failure never becomes PASS.
"""
from pathlib import Path
import faulthandler
import argparse,datetime,hashlib,json,os,sys,time,traceback
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['preview','train','evaluate'],required=True)
p.add_argument('--parent',default='/data/code/ouyanghao/oyh/phygile-reproduction-20261017')
p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
p.add_argument('--motion-file');p.add_argument('--num-envs',type=int,default=8)
p.add_argument('--iterations',type=int,default=2);p.add_argument('--steps',type=int,default=100)
p.add_argument('--seed',type=int,default=2026);p.add_argument('--viewer-approval')
p.add_argument('--wait-viewer-seconds',type=int,default=0)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
root=Path(__file__).resolve().parents[1];parent=Path(a.parent);out=Path(a.output)
out.mkdir(parents=True,exist_ok=False)
if a.mode=='train' and (a.headless or not a.enable_cameras):raise RuntimeError('New RL training requires a visible camera-enabled GUI')
checkpoint=Path(a.checkpoint).resolve()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
receipt={'status':'STARTING','mode':a.mode,'training':a.mode=='train','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'parent_checkpoint':str(checkpoint),'parent_sha256':sha(checkpoint),'device':a.device,'headless':a.headless,'display':os.environ.get('DISPLAY'),'seed':a.seed,'num_envs':a.num_envs,'source_sha256':sha(__file__),'task':'Unitree-G1-PhyGile-v0','paper_benchmark_reproduced':False}
def save():
    temp=out/'receipt.partial';temp.write_text(json.dumps(receipt,indent=2)+'\n');temp.replace(out/'receipt.json')
save();app=None;env=None
stack_file=(out/'startup_stack.txt').open('w')
faulthandler.dump_traceback_later(120,repeat=True,file=stack_file)
try:
    app=AppLauncher(a).app
    faulthandler.cancel_dump_traceback_later()
    print('NATIVE_GUI_APP_LAUNCHED',flush=True)
    import carb,gymnasium as gym,numpy as np,torch
    from PIL import Image
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatWoStateEstimationEnvCfg
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner
    from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
    from lab_task import PhyGileUnitreeEnvCfg,semantic_configuration,register
    torch.set_num_threads(4);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    if a.mode!='train':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        receipt['inference_tf32']=False
    torch.cuda.set_per_process_memory_fraction(.20,torch.device(a.device))
    cfg=PhyGileUnitreeEnvCfg();original=G1FlatWoStateEstimationEnvCfg()
    eq=semantic_configuration(cfg)==semantic_configuration(original)
    if not eq:raise RuntimeError('Adapted MDP differs from the original configuration')
    receipt.update(configuration_semantic_equality=True,environment_mro=[c.__module__+'.'+c.__name__ for c in type(cfg).__mro__])
    cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device=a.device
    cfg.commands.motion.debug_vis=False;cfg.scene.contact_forces.debug_vis=False
    if a.motion_file:
        cfg.commands.motion.motion_dir=None;cfg.commands.motion.motion_file=str(Path(a.motion_file).resolve())
    else:
        cfg.commands.motion.motion_file=None;cfg.commands.motion.motion_dir=str(parent/'data/train')
    cfg.viewer.eye=(3.,-3.,2.);cfg.viewer.lookat=(0.,0.,.8);cfg.viewer.resolution=(960,540)
    cfg.scene.terrain.visual_material=None
    if a.mode!='train':
        cfg.observations.policy.enable_corruption=False;cfg.events.push_robot=None
        cfg.commands.motion.random_static_prob=0
        cfg.events.physics_material=None;cfg.events.add_joint_default_pos=None;cfg.events.base_com=None
        cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={};cfg.commands.motion.joint_position_range=(0.,0.)
        receipt['evaluation_overrides']=['no observation noise','no startup material/default-joint/COM randomization','no pushes','zero reset pose/velocity/joint perturbation']
    if a.mode=='evaluate':
        if a.num_envs!=1 or not a.motion_file:raise ValueError('Evaluation requires exactly one environment and a motion file')
        with np.load(a.motion_file,allow_pickle=False) as data:
            nframes=len(data['joint_pos']);fps=float(np.asarray(data['fps']).reshape(-1)[0])
        if abs(cfg.sim.dt*cfg.decimation-1/fps)>1e-8:raise RuntimeError('Reference/control rate mismatch')
        cfg.episode_length_s=nframes/fps;cfg.commands.motion.curriculum_enabled=False
    register();env=gym.make('Unitree-G1-PhyGile-v0',cfg=cfg,render_mode='rgb_array')
    base=env.unwrapped;robot=base.scene['robot'];mt=base.command_manager.get_term('motion')
    if a.mode=='evaluate':
        def reset_start(ids):mt.time_steps[ids]=0
        mt._adaptive_sampling=reset_start
    wrapped=RslRlVecEnvWrapper(env)
    agent=G1FlatPPORunnerCfg();agent.device=a.device;agent.seed=a.seed;agent.logger='tensorboard'
    agent.experiment_name='unitree_phygile_adaptation';agent.save_interval=100000
    runner=MotionOnPolicyRunner(wrapped,agent.to_dict(),log_dir=str(out/'training') if a.mode=='train' else None,device=a.device,upload_models_to_wandb=False)
    runner.load(str(checkpoint),load_optimizer=a.mode=='train')
    loaded=torch.load(checkpoint,map_location=a.device,weights_only=False)
    for key,value in runner.alg.policy.state_dict().items():
        if not torch.equal(value,loaded['model_state_dict'][key]):raise RuntimeError('Policy state load mismatch: '+key)
    for name,key in [('obs_normalizer','obs_norm_state_dict'),('privileged_obs_normalizer','privileged_obs_norm_state_dict')]:
        for k,v in getattr(runner,name).state_dict().items():
            if not torch.equal(v,loaded[key][k]):raise RuntimeError('Normalizer mismatch: '+k)
    if a.mode=='train':
        runner.current_learning_iteration=int(loaded['iter'])+1
        rates={g['lr'] for g in runner.alg.optimizer.param_groups}
        if len(rates)!=1:raise RuntimeError('Inconsistent optimizer LR')
        runner.alg.learning_rate=next(iter(rates))
    obs,_=wrapped.get_observations()
    policy=runner.get_inference_policy(device=a.device)
    probe=policy(obs)
    if not torch.isfinite(probe).all():raise RuntimeError('Nonfinite restored policy action')
    def values(t):return t[0].detach().cpu().tolist()
    action_term=next(iter(base.action_manager._terms.values()))
    contract={'joint_names':robot.joint_names,'body_names':robot.body_names,'tracked_body_names':mt.cfg.body_names,'anchor_body_name':mt.cfg.anchor_body_name,
      'joint_sdk_names':cfg.scene.robot.joint_sdk_names,'physics_dt':base.physics_dt,'control_dt':base.step_dt,'decimation':cfg.decimation,
      'default_joint_pos':values(robot.data.default_joint_pos),'stiffness':values(robot.data.default_joint_stiffness),'damping':values(robot.data.default_joint_damping),
      'action_scale':values(action_term._scale),'action_offset':values(action_term._offset),'joint_effort_limits':values(robot.data.joint_effort_limits),
      'joint_armature':values(robot.data.default_joint_armature),'observation_names':base.observation_manager.active_terms,
      'observation_dims':{k:list(v) for k,v in base.observation_manager.group_obs_dim.items()},'future_steps':mt.cfg.future_steps,'encoder_future_steps':mt.cfg.encoder_future_steps,'encoder_future_stride':mt.cfg.encoder_future_stride,
      'reference_fps':float(np.asarray(mt.motion.fps).reshape(-1)[0]) if hasattr(mt.motion,'fps') else 50.,'policy_config':agent.policy.to_dict(),'checkpoint_sha256':receipt['parent_sha256'],
      'selected_policy':runner.alg.policy.__class__.__module__+'.'+runner.alg.policy.__class__.__name__,'selected_algorithm':runner.alg.__class__.__module__+'.'+runner.alg.__class__.__name__,
      'robot_asset':str(cfg.scene.robot.spawn.asset_path),'active_experts':runner.alg.policy._max_active_experts,
      'body_mass':robot.root_physx_view.get_masses()[0].cpu().tolist(),'body_inertia':robot.root_physx_view.get_inertias()[0].cpu().tolist()}
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    export_deploy_cfg(base,str(out/'unitree_export'))
    synthetic=torch.randn((5,obs.shape[-1]),generator=torch.Generator(device='cpu').manual_seed(42)).to(a.device)
    with torch.inference_mode():synthetic_actions=policy(synthetic)
    np.savez_compressed(out/'synthetic_policy_fixture.npz',observations=synthetic.cpu().numpy(),actions=synthetic_actions.cpu().numpy())
    receipt.update(policy_loaded_exactly=True,normalizers_loaded_exactly=True,policy_observation_shape=list(obs.shape),action_shape=list(probe.shape),policy=contract['selected_policy'],algorithm=contract['selected_algorithm'],active_experts=contract['active_experts'],status='GUI_READY')
    carb.settings.get_settings().set('/app/window/title','PhyGile / Unitree RL Lab / '+a.mode)
    frame=env.render()
    if frame is None or np.asarray(frame).size==0:raise RuntimeError('Actual simulator render is empty')
    Image.fromarray(np.asarray(frame)[...,:3]).save(out/'frame.png')
    receipt['render_shape']=list(frame.shape);save()
    if a.mode=='train':
        approval=Path(a.viewer_approval or '')
        if not approval.is_file():raise RuntimeError('Missing prior local viewer acceptance receipt')
        verified=json.loads(approval.read_text())
        if verified.get('status')!='LOCAL_VIEWER_VERIFIED':raise RuntimeError('Viewer acceptance was not verified')
    if a.wait_viewer_seconds:
        end=time.monotonic()+a.wait_viewer_seconds
        while time.monotonic()<end:app.update();time.sleep(.05)
    # Keep the actual training/evaluation render available through the viewer portal.
    original_step=wrapped.step;counter=[0]
    def visible_step(actions):
        result=original_step(actions);counter[0]+=1
        if counter[0]%4==0:
            frame=env.render()
            if frame is not None:
                path=out/'frame.next.png';Image.fromarray(np.asarray(frame)[...,:3]).save(path);path.replace(out/'frame.png')
        return result
    wrapped.step=visible_step
    if a.mode=='train':
        before={k:v.detach().clone() for k,v in runner.alg.policy.named_parameters()}
        initial_steps=sorted({float(s['step']) for s in runner.alg.optimizer.state.values() if 'step' in s})
        calls=[0]
        def count_step(*_):calls[0]+=1
        runner.alg.optimizer.register_step_post_hook(count_step)
        runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=True)
        final=out/'adapted_checkpoint.pt';runner.save(str(final))
        after=dict(runner.alg.policy.named_parameters());changed=[k for k,v in before.items() if not torch.equal(v,after[k])]
        if not changed or not all(torch.isfinite(v).all() for v in after.values()):raise RuntimeError('No finite policy training update observed')
        expected=a.iterations*agent.algorithm.num_learning_epochs*agent.algorithm.num_mini_batches
        if calls[0]!=expected:raise RuntimeError('Actual Adam calls differ from bounded PPO budget')
        receipt.update(status='BOUNDED_TRAINING_COMPLETED',parent_ppo_iterations=int(loaded['iter'])+1,additional_ppo_iterations=a.iterations,
          final_checkpoint=str(final),final_sha256=sha(final),changed_policy_tensors=len(changed),adam_steps_before=initial_steps,
          adam_steps_after=sorted({float(s['step']) for s in runner.alg.optimizer.state.values() if 'step' in s}),actual_adam_calls=calls[0],control_calls=counter[0])
    elif a.mode=='evaluate':
        from isaaclab.utils.math import yaw_quat,quat_mul,quat_inv,quat_apply
        base.reset(seed=a.seed)
        if not bool(torch.all(mt.time_steps==0)):raise RuntimeError('Evaluation did not start at frame 0')
        anchor=mt.anchor_pos_w[:,None,:].repeat(1,len(mt.cfg.body_names),1)
        aq=mt.anchor_quat_w[:,None,:].repeat(1,len(mt.cfg.body_names),1)
        rq=mt.robot_anchor_quat_w[:,None,:].repeat(1,len(mt.cfg.body_names),1)
        pos=mt.robot_anchor_pos_w[:,None,:].repeat(1,len(mt.cfg.body_names),1);pos[...,2]=anchor[...,2]
        rotation=yaw_quat(quat_mul(rq,quat_inv(aq)))
        mt.body_quat_relative_w=quat_mul(rotation,mt.body_quat_w)
        mt.body_pos_relative_w=pos+quat_apply(rotation,mt.body_pos_w-anchor)
        base.obs_buf=base.observation_manager.compute();obs,_=wrapped.get_observations()
        np.savez_compressed(out/'policy_io_private.npz',observations=obs.detach().cpu().numpy(),actions=policy(obs).detach().cpu().numpy())
        rows=[];original_compute=base.termination_manager.compute
        def capture():
            result=original_compute()
            rows.append({'frame':int(mt.time_steps[0]),'reference_body_pos':mt.body_pos_w[0].cpu().numpy().copy(),
              'robot_body_pos':mt.robot_body_pos_w[0].cpu().numpy().copy(),'reference_joint_pos':mt.joint_pos[0].cpu().numpy().copy(),
              'robot_joint_pos':mt.robot_joint_pos[0].cpu().numpy().copy(),'reference_joint_vel':mt.joint_vel[0].cpu().numpy().copy(),
              'robot_joint_vel':mt.robot_joint_vel[0].cpu().numpy().copy(),'terminated':bool(base.termination_manager.terminated[0]),'timeout':bool(base.termination_manager.time_outs[0])})
            return result
        base.termination_manager.compute=capture
        with torch.inference_mode():
            for _ in range(nframes+2):
                actions=policy(obs)
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite action')
                obs,_,dones,_=wrapped.step(actions)
                if bool(dones[0]):break
        raw={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        if not np.array_equal(raw['frame'],np.arange(len(rows))):raise RuntimeError('Reference frame continuity mismatch')
        if not bool(raw['terminated'][-1] or raw['timeout'][-1]):raise RuntimeError('First episode did not finish')
        for k,v in raw.items():
            if not np.isfinite(v).all():raise RuntimeError('Nonfinite raw state: '+k)
        np.savez_compressed(out/'raw_pre_reset.npz',**raw)
        receipt.update(status='FIRST_EPISODE_EVALUATED',evaluation_type='simulation_only',optimizer_steps=0,
          motion_frames=nframes,motion_sha256=sha(a.motion_file),frames_evaluated=len(rows),terminated=bool(raw['terminated'][-1]),timeout=bool(raw['timeout'][-1]),
          full_clip_success=bool(len(rows)==nframes and not raw['terminated'].any()),
          metrics={'mpjpe_m_until_termination':float(np.linalg.norm(raw['reference_body_pos']-raw['robot_body_pos'],axis=-1).mean()),
           'mpjae_rad_until_termination':float(np.abs(raw['reference_joint_pos']-raw['robot_joint_pos']).mean()),
           'mpjve_rad_s_until_termination':float(np.abs(raw['reference_joint_vel']-raw['robot_joint_vel']).mean())},
          control_calls=counter[0],physics_steps=counter[0]*cfg.decimation,protocol='First episode from frame 0; pre-reset/post-physics state compared with current reference t; deterministic reset; original failure terminations retained. Errors are censored at failure.')
    else:
        with torch.inference_mode():
            for _ in range(a.steps):
                actions=policy(obs)
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite action')
                obs,_,_,_=wrapped.step(actions)
        receipt.update(status='RENDERED_POLICY_PREVIEW_COMPLETED',control_calls=counter[0])
    receipt.update(parent_unchanged=sha(checkpoint)==receipt['parent_sha256'],finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(a.device)))
    if not receipt['parent_unchanged']:raise RuntimeError('Parent checkpoint changed')
    save();print('UNITREE_PHYGILE_RESULT',json.dumps(receipt),flush=True)
except BaseException as exc:
    receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());save();raise
finally:
    if env is not None:env.close()
    if app is not None:app.close()
