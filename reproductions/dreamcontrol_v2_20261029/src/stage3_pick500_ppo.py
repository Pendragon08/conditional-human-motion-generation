"""Independent Pick PPO with the generated500-reference/object-bound scene.

Uses the established task and PPO configuration, retaining the old2-reference
baseline. Integration tests are explicitly bounded and cannot start a formal run.
"""
import argparse,json,os,math,shutil,time,traceback
from pathlib import Path
from isaaclab.app import AppLauncher

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference-manifest',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--integration-test',action='store_true')
p.add_argument('--capacity-test',action='store_true')
p.add_argument('--num-envs',type=int,default=8912)
p.add_argument('--max-iterations',type=int,default=3000)
p.add_argument('--steps',type=int,default=128)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
assert not (a.integration_test and a.capacity_test)
a.output_dir.mkdir(exist_ok=False)
report={'status':'preflight','pid':os.getpid(),'policy_updates':0,'num_envs':a.num_envs,
    'integration_test':a.integration_test,'capacity_test':a.capacity_test,'formal_training':not(a.integration_test or a.capacity_test)}
def save():
 t=a.output_dir/'status.tmp';t.write_text(json.dumps(report,indent=2,allow_nan=False));t.replace(a.output_dir/'status.json')
save();app=env=None;began=time.monotonic()
try:
    import hashlib
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    formal=report['formal_training'];manifest=json.loads(a.reference_manifest.read_text());manifest_sha=sha(a.reference_manifest)
    if a.integration_test:assert a.num_envs==4 and a.max_iterations==2
    else:assert len(manifest['references'])==500 and a.num_envs==8912
    prior=R/'stage2_prior29_20261103/runs/grab_omni29_mdm_baseline_v1/status.json'
    prior_status=json.loads(prior.read_text());assert prior_status['status']=='completed' and prior_status['cursor']['step']==10000
    floor=8*1024**3+456*1024**2
    if formal:
        assert a.max_iterations==3000
        audit=json.loads((O/'final_library_audit.json').read_text())
        assert audit['status']=='passed' and audit['reference_count']==500 and audit['manifest_sha256']==manifest_sha
        smoke=json.loads((O/'ppo_integration_v1/status.json').read_text())
        assert smoke['status']=='ppo_completed' and smoke['policy_updates']==2 and smoke['checkpoint_roundtrip_verified']
        capacity=json.loads((O/'ppo_capacity8912_v1/status.json').read_text())
        assert capacity['status']=='capacity_completed' and capacity['reference_manifest_sha256']==manifest_sha
        assert capacity['num_envs']==8912 and capacity['steps']>=128
        size=smoke['largest_checkpoint_bytes'];checkpoint_budget=math.ceil(size*1.25/1024**2)*1024**2
        floor=max(11*1024**3,8*1024**3+356*1024**2+63*checkpoint_budget)
    assert shutil.disk_usage('/data').free>=floor,'existing disk/checkpoint reserve is not satisfied'
    report.update(required_start_free_bytes=floor,reference_manifest_sha256=manifest_sha,reference_count=len(manifest['references']));save()
    app=AppLauncher(a).app
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from stage3_pick500_env import Pick500EnvCfg
    from stage3_pick29_env import live_observation,ASSET,SCENE_SHA
    from stage3_pick29_core import BODY_JOINTS
    torch.manual_seed(20261103);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12,device=a.device)
    cfg=Pick500EnvCfg();cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=20261103
    cfg.reference_manifest=str(a.reference_manifest)
    if a.integration_test:cfg.minimum_reference_count=2
    env=ManagerBasedRLEnv(cfg=cfg);obs,_=env.reset()
    assert obs['policy'].shape==(a.num_envs,102) and env.action_manager.total_action_dim==31
    robot=env.scene['robot'];assert robot.find_joints(BODY_JOINTS,preserve_order=True)[1]==BODY_JOINTS
    selected=env._reference_ids.clone();expected=env._reference_bank.object_positions[selected]+env.scene.env_origins
    object_error=float((env.scene['object'].data.root_pos_w-expected).abs().max());assert object_error<1e-5
    before=live_observation(env).clone();env._reference_ids=(selected+1)%env._reference_bank.count
    leak=float((live_observation(env)-before).abs().max());env._reference_ids=selected;assert leak==0
    report.update(status='environment_ready',object_reset_error_m=object_error,actor_reference_change_error=leak,
        body_dof=29,actions=31,actor_observation=102,asset_sha256=sha(ASSET),scene_sha256=SCENE_SHA);save()
    if a.capacity_test:
        for _ in range(a.steps):
            action=torch.zeros(a.num_envs,31,device=a.device);action[:,-2:]=1
            obs,reward,term,trunc,info=env.step(action)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        report.update(status='capacity_completed',steps=a.steps,reference_manifest_sha256=manifest_sha)
    else:
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
        runner_cfg=G1FlatPPORunnerCfg();runner_cfg.max_iterations=a.max_iterations
        runner_cfg.experiment_name='dreamcontrol_v2_pick29_reference500';runner_cfg.seed=20261103
        assert runner_cfg.num_steps_per_env==24
        runner=OnPolicyRunner(RslRlVecEnvWrapper(env),runner_cfg.to_dict(),log_dir=str(a.output_dir/'ppo'),device=a.device)
        provenance={'asset_sha256':sha(ASSET),'scene_sha256':SCENE_SHA,'reference_bank_sha256':manifest_sha,
            'reference_count':env._reference_bank.count,'diffusion_checkpoint_sha256':prior_status['last_checkpoint']['sha256'],
            'body_joint_names':BODY_JOINTS,'policy_actions':31,'policy_observations':102,'actor_reference_input':False,
            'num_envs':a.num_envs,'max_iterations':a.max_iterations,'runner_config':runner_cfg.to_dict(),
            'test_scope':a.integration_test,'object_pose_policy':'per_reference_fixed',
            'source_sha256':{n:sha(Path(__file__).parent/n) for n in ['stage3_pick500_ppo.py','stage3_pick500_env.py','stage3_pick29_env.py','stage3_pick29_core.py']}}
        (a.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2))
        original_save=runner.save
        def save_bound(path,infos=None):
            assert not Path(path).exists(),'checkpoint overwrite refused'
            return original_save(path,infos={'provenance':provenance,'user_infos':infos})
        runner.save=save_bound;original_log=runner.log
        def log_progress(locs,width=80,pad=35):
            original_log(locs,width=width,pad=pad)
            metrics={'iteration_index':int(locs['it']),'policy_updates':int(locs['it'])+1,
                'collection_time_s':float(locs['collection_time']),'learn_time_s':float(locs['learn_time'])}
            metrics.update({k:float(v) for k,v in locs.get('loss_dict',{}).items()})
            assert all(torch.isfinite(t).all() for t in runner.alg.policy.parameters())
            with (a.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            report.update(policy_updates=metrics['policy_updates'],last_metrics=metrics);save()
        runner.log=log_progress;original_step=env.step
        def guarded_step(action):
            if env.common_step_counter%100==0:
                assert shutil.disk_usage('/data').free>=8*1024**3+356*1024**2,'running disk reserve'
            return original_step(action)
        env.step=guarded_step;report.update(status='ppo_running',max_iterations=a.max_iterations);save()
        runner.learn(num_learning_iterations=a.max_iterations,init_at_random_ep_len=False)
        final=a.output_dir/('ppo/model_'+str(a.max_iterations-1)+'.pt')
        saved=torch.load(final,map_location='cpu',weights_only=True)
        assert saved['iter']==a.max_iterations-1 and saved['infos']['provenance']['reference_bank_sha256']==manifest_sha
        if a.integration_test:
            current=runner.alg.policy.state_dict()
            assert all(torch.equal(saved['model_state_dict'][k],v.cpu()) for k,v in current.items())
            assert saved['optimizer_state_dict']['state']
        report.update(status='ppo_completed',policy_updates=a.max_iterations,checkpoint_roundtrip_verified=True,
            final_checkpoint=str(final),final_checkpoint_sha256=sha(final),largest_checkpoint_bytes=max(x.stat().st_size for x in (a.output_dir/'ppo').glob('*.pt')),
            observed_task_success_claimed=False)
    report['elapsed_seconds']=time.monotonic()-began;save();print(json.dumps(report),flush=True)
except BaseException as e:
    report.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
    if env is not None:env.close()
    if app is not None:app.close(wait_for_replicator=False)
