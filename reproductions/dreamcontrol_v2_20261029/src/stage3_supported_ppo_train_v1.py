"""Task-specific autonomous PPO, real capacity and read-only100-env evaluation."""
import argparse,hashlib,json,math,os,shutil,sys,time,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103'
sys.path[:0]=[str(B/'src'),str(S/'src')]
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--task',choices=['pour','bimanual_box_lift','box_push_pull'],required=True)
p.add_argument('--mode',choices=['integration','capacity','formal','evaluate'],required=True)
p.add_argument('--reference-manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=8912);p.add_argument('--max-iterations',type=int,default=5000)
p.add_argument('--training-run',type=Path);p.add_argument('--ready-report',type=Path)
AppLauncher.add_app_launcher_args(p);a=p.parse_args();a.output_dir.mkdir(exist_ok=False)
state={'status':'preflight','pid':os.getpid(),'task':a.task,'mode':a.mode,'policy_updates':0,
       'formal_training_started':False,'object_paths_injected':False};began=time.monotonic();app=env=None
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():
    state['elapsed_seconds']=time.monotonic()-began;t=a.output_dir/'status.tmp'
    t.write_text(json.dumps(state,indent=2,allow_nan=False));t.replace(a.output_dir/'status.json')
save()
try:
    m=read(a.reference_manifest);msha=sha(a.reference_manifest)
    assert m['task']==a.task
    if a.mode=='integration':assert a.num_envs==4 and a.max_iterations==2 and len(m['references'])==2
    else:assert len(m['references'])==500
    if a.mode in ['formal','capacity']:assert a.num_envs==8912
    if a.mode=='evaluate':assert a.num_envs==100 and a.training_run
    if a.mode=='formal':
        ready=read(a.ready_report);assert ready['status']=='task_ppo_formal_preflight_passed'
        assert ready['task']==a.task and ready['manifest_sha256']==msha and ready['iteration_target']==a.max_iterations
        assert sha(ready['capacity_status'])==ready['capacity_status_sha256']
        assert read(ready['capacity_status'])['status']=='capacity_completed'
        assert shutil.disk_usage('/data').free>=ready['required_start_free_bytes']>=11*1024**3
    else:assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2
    app=AppLauncher(a).app
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from stage3_supported_ppo_env_v1 import make_cfg,measured
    from stage3_pick29_env import ASSET,live_observation
    from stage3_pick29_core import BODY_JOINTS
    torch.set_num_threads(2);torch.manual_seed(20261105)
    torch.cuda.set_per_process_memory_fraction(.12,device=a.device)
    cfg=make_cfg(a.task,a.reference_manifest,a.device,a.num_envs,2 if a.mode=='integration' else 500,a.mode=='evaluate')
    env=ManagerBasedRLEnv(cfg=cfg);wrapped=RslRlVecEnvWrapper(env);obs,_=wrapped.reset()
    assert obs.shape==(a.num_envs,102) and env.action_manager.total_action_dim==31
    assert env.scene['robot'].find_joints(BODY_JOINTS,preserve_order=True)[1]==BODY_JOINTS
    selected=env._reference_ids.clone();live=live_observation(env).clone()
    env._reference_ids=(selected+1)%env._reference_bank.count
    leak=float((live_observation(env)-live).abs().max());env._reference_ids=selected;assert leak==0
    expected=env._reference_bank.object_positions[selected]+env.scene.env_origins
    reset_error=float((env.scene['object'].data.root_pos_w-expected).abs().max()) if a.mode!='evaluate' else None
    if reset_error is not None:assert reset_error<1e-5
    contract={'task':a.task,'asset_sha256':sha(ASSET),'reference_manifest_sha256':msha,
              'reference_count':env._reference_bank.count,'generation_scene_sha256':m['scene_sha256'],
              'object_mass_kg':.1 if a.task=='pour' else 1.,'physics':'PhysX dynamic/gravity/contact-driven object',
              'actor_observation':102,'policy_actions':31,'body_dof':29,'actor_reference_input':False,
              'num_envs':a.num_envs,'max_iterations':a.max_iterations,
              'scope':'project bounded tabletop proxy task; Pour measures bottle cycle, not liquid transfer',
              'source_sha256':{n:sha(B/'src'/n) for n in ['stage3_supported_ppo_train_v1.py','stage3_supported_ppo_env_v1.py','stage3_supported_ppo_core_v1.py']}}
    state.update(status='environment_ready',reference_count=env._reference_bank.count,
                 manifest_sha256=msha,actor_reference_mutation_difference=leak,
                 object_reset_error_m=reset_error,contract=contract);save()
    runner_cfg=G1FlatPPORunnerCfg();runner_cfg.max_iterations=a.max_iterations
    runner_cfg.experiment_name='dcv2_'+a.task+'_reference500';runner_cfg.seed=20261105
    assert runner_cfg.num_steps_per_env==24
    if a.mode=='capacity':
        for step in range(128):
            action=torch.zeros(a.num_envs,31,device=a.device);action[:,-2:]=1
            obs,reward,done,extras=wrapped.step(action)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            if step==0:
                contacts=measured(env)['contacts'];assert contacts.shape==(a.num_envs,2)
        state.update(status='capacity_completed',steps=128,physical_joint_count=len(env.scene['robot'].joint_names));save()
    elif a.mode=='evaluate':
        training=read(a.training_run/'status.json');assert training['status']=='ppo_completed'
        checkpoint=Path(training['final_checkpoint']);before=sha(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['infos']['provenance']['task']==a.task
        assert saved['infos']['provenance']['reference_manifest_sha256']==msha
        runner=OnPolicyRunner(wrapped,runner_cfg.to_dict(),log_dir=None,device=a.device)
        runner.load(str(checkpoint),load_optimizer=False);actor=runner.get_inference_policy(device=a.device)
        def policy_hash():
            h=hashlib.sha256()
            for k,v in sorted(runner.alg.policy.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        policy_before=policy_hash();active=torch.ones(a.num_envs,device=a.device,dtype=torch.bool)
        outcomes={k:torch.zeros_like(active) for k in env._milestones}
        positions=(env.scene['object'].data.root_pos_w-env.scene.env_origins).clone()
        for step in range(500):
            with torch.inference_mode():obs,reward,done,extras=wrapped.step(actor(obs))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            for key in outcomes:
                value=torch.where(done.bool(),env._last_episode[key],env._milestones[key])
                outcomes[key]|=active&value
            active&=~done.bool()
            if not active.any():break
        assert policy_hash()==policy_before and sha(checkpoint)==before
        report={'status':'task_ppo_evaluation_completed','task':a.task,'num_envs':100,
                'seed':20261105,'new_object_xy_distribution':'uniform nominal+-0.025m',
                'success_count':int(outcomes['success'].sum()),'success_ratio':float(outcomes['success'].float().mean()),
                'milestone_counts':{k:int(v.sum()) for k,v in outcomes.items()},
                'success_definition':{'pour':'contact/lift actual bottle, tilt>60deg, restore upright and place/release; no liquid model',
                  'bimanual_box_lift':'both hands object-contact, actual lift>0.10m/carry>0.08m/place/release',
                  'box_push_pull':'actual push>0.08m, release, recontact, return within0.025m'}[a.task],
                'manifest_sha256':msha,'checkpoint_sha256':before,'policy_updates':0,
                'policy_unchanged':True,'initial_object_positions':positions.cpu().tolist(),'steps':step+1,
                'contract':contract,'training_budget_complete_is_not_task_success':True}
        (a.output_dir/'report.json').open('x').write(json.dumps(report,indent=2,allow_nan=False))
        state.update(report);save()
    else:
        runner=OnPolicyRunner(wrapped,runner_cfg.to_dict(),log_dir=str(a.output_dir/'ppo'),device=a.device)
        (a.output_dir/'provenance.json').open('x').write(json.dumps(contract,indent=2))
        original_save=runner.save
        def save_bound(path,infos=None):
            assert not Path(path).exists(),'checkpoint overwrite refused'
            return original_save(path,infos={'provenance':contract,'user_infos':infos})
        runner.save=save_bound;original_log=runner.log
        def log(locs,width=80,pad=35):
            original_log(locs,width=width,pad=pad)
            value={'iteration':int(locs['it']),'policy_updates':int(locs['it'])+1,
                   'collection_time_s':float(locs['collection_time']),'learn_time_s':float(locs['learn_time'])}
            value.update({k:float(v) for k,v in locs.get('loss_dict',{}).items()})
            assert all(torch.isfinite(v).all() for v in runner.alg.policy.parameters())
            with (a.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
            state.update(status='ppo_running',policy_updates=value['policy_updates'],
                         formal_training_started=a.mode=='formal',last_metrics=value);save()
        runner.log=log;state.update(status='ppo_process_started');save()
        runner.learn(num_learning_iterations=a.max_iterations,init_at_random_ep_len=False)
        checkpoint=a.output_dir/('ppo/model_'+str(a.max_iterations-1)+'.pt')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['iter']==a.max_iterations-1 and saved['infos']['provenance']['task']==a.task
        assert saved['optimizer_state_dict']['state']
        assert all(torch.equal(v.cpu(),saved['model_state_dict'][k]) for k,v in runner.alg.policy.state_dict().items())
        current=runner.alg.optimizer.state_dict() if hasattr(runner.alg,'optimizer') else None
        def same(left,right):
            if isinstance(left,torch.Tensor):
                return bool(torch.isfinite(left).all()) and torch.equal(left.cpu(),right.cpu())
            if isinstance(left,dict):return left.keys()==right.keys() and all(same(left[k],right[k]) for k in left)
            if isinstance(left,(tuple,list)):return len(left)==len(right) and all(same(x,y) for x,y in zip(left,right))
            return left==right
        assert current is not None and same(current,saved['optimizer_state_dict'])
        state.update(status='ppo_completed',policy_updates=a.max_iterations,checkpoint_roundtrip_verified=True,
                     final_checkpoint=str(checkpoint),final_checkpoint_sha256=sha(checkpoint),
                     largest_checkpoint_bytes=max(x.stat().st_size for x in (a.output_dir/'ppo').glob('*.pt')),
                     observed_task_success_claimed=False);save()
    print(json.dumps(state),flush=True)
except BaseException as error:
    state.update(status='failed_preserved',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
    if env is not None:env.close()
    if app is not None:app.close(wait_for_replicator=False)
