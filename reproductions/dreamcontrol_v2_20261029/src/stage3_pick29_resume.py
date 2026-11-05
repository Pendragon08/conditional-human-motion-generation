"""Resume the existing Pick policy to the user's new total budget, preserving saved training state."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import shutil
import math

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode',choices=['smoke','ppo'],default='smoke')
parser.add_argument('--num-envs',type=int,default=4)
parser.add_argument('--steps',type=int,default=100)
parser.add_argument('--max-iterations',type=int,default=3000)
parser.add_argument('--resume',type=Path,required=True)
parser.add_argument('--verify-resume-only',action='store_true')
parser.add_argument('--reference-manifest',default='')
parser.add_argument('--output-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
args.output_dir.mkdir(exist_ok=False)
report={'pid':os.getpid(),'mode':args.mode,'num_envs':args.num_envs,'policy_updates':0,
        'status':'starting_simulation','reference_manifest':args.reference_manifest}


def save():
    tmp=args.output_dir/'status.tmp'
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False))
    tmp.replace(args.output_dir/'status.json')


save()
app=None
env=None
try:
    if args.mode=='ppo' and not args.reference_manifest:
        raise ValueError('PPO requires hash-bound generated Pick references; smoke is allowed without them')
    if args.mode=='ppo':
        stage=args.output_dir.parents[1]
        prior=stage.parent/'stage2_prior29_20261103/runs/grab_omni29_mdm_baseline_v1/status.json'
        prior_status=json.loads(prior.read_text())
        assert prior_status['status']=='completed' and prior_status['cursor']['step']==10000
        need=11*1024**3
        if args.max_iterations>2 and not args.verify_resume_only:
            small=json.loads((stage/'reports/ppo_smoke_4/status.json').read_text())
            capacity=json.loads((stage/'reports/physx_capacity_8912/status.json').read_text())
            assert small['status']=='ppo_completed' and small['policy_updates']==2
            assert capacity['status']=='smoke_completed' and capacity['num_envs']==8912
            assert args.num_envs==8912 and args.max_iterations==3000
            sizes=[p.stat().st_size for p in (stage/'reports/ppo_smoke_4/ppo').glob('model_*.pt')]
            checkpoint_budget=math.ceil(max(sizes)*1.25/1024**2)*1024**2
            need=max(need,8*1024**3+356*1024**2+(math.ceil(args.max_iterations/50)+3)*checkpoint_budget)
        assert shutil.disk_usage('/data').free>=need,'PPO/checkpoint disk budget'
        report['required_free_bytes_at_launch']=need
    began=time.monotonic()
    app=AppLauncher(args).app
    import torch
    import isaaclab.utils.math as math_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from stage3_pick29_env import Pick29EnvCfg,live_observation,ASSET,SCENE_SHA
    from stage3_pick29_core import BODY_JOINTS,sha
    torch.manual_seed(20261103)
    torch.cuda.set_per_process_memory_fraction(.12,device=args.device)
    cfg=Pick29EnvCfg()
    cfg.scene.num_envs=args.num_envs
    cfg.sim.device=args.device
    cfg.seed=20261103
    cfg.reference_manifest=args.reference_manifest
    env=ManagerBasedRLEnv(cfg=cfg)
    obs,_=env.reset()
    robot=env.scene['robot']
    body=robot.find_joints(BODY_JOINTS,preserve_order=True)
    assert body[1]==BODY_JOINTS and len(body[0])==29
    assert env.action_manager.total_action_dim==31
    assert obs['policy'].shape==(args.num_envs,102)
    # Two hand commands must actually produce distinct joint target vectors.
    action=torch.zeros((args.num_envs,31),device=env.device)
    action[:,-2:]=1
    env.action_manager.process_action(action)
    opened={n:env.action_manager.get_term(n).processed_actions.clone() for n in ('left_hand','right_hand')}
    action[:,-2:]=-1
    env.action_manager.process_action(action)
    hand_delta={n:float((env.action_manager.get_term(n).processed_actions-opened[n]).abs().max())
                for n in opened}
    assert min(hand_delta.values())>1
    # Mutating private reference selection cannot change actor input.
    before=live_observation(env).clone()
    old=env._reference_ids.clone()
    env._reference_ids[:]=1234567
    after=live_observation(env)
    env._reference_ids[:]=old
    assert torch.equal(before,after)
    if args.mode=='smoke':
        import sys
        import numpy as np
        sys.path.insert(0,str(ASSET.parents[2]/'training_disk_guard_20261101'))
        from g1_fk import URDFFK,PAPER_22_LINKS
        urdf=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
        fk=URDFFK(urdf)
        frames=env._keypoint_frames
        geometry_errors=[];geometry_rotation_errors=[]
        for roll,pitch in [(0.,0.),(.15,-.12),(-.17,.13)]:
            q=robot.data.default_joint_pos.clone()
            q[:,robot.joint_names.index('waist_roll_joint')]=roll
            q[:,robot.joint_names.index('waist_pitch_joint')]=pitch
            robot.write_joint_state_to_sim(q,torch.zeros_like(q))
            env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
            rp=robot.data.root_pos_w.cpu().numpy();rq=robot.data.root_quat_w.cpu().numpy()
            points,rot=fk.forward(rp,rq[:,[1,2,3,0]],q[:,body[0]].cpu().numpy(),BODY_JOINTS,return_rotations=True)
            geometry_errors.append(float(np.abs(frames.positions(robot.data.body_pos_w,robot.data.body_quat_w).cpu().numpy()-points).max()))
            actual=(math_utils.matrix_from_quat(robot.data.body_quat_w[:,frames.indices])@frames.rotations).cpu().numpy()
            geometry_rotation_errors.append(float(np.abs(actual-rot).max()))
        env.reset()
        limits=robot.data.joint_pos_limits[0]
        soft=robot.data.soft_joint_pos_limits[0]
        initial_object_height=(env.scene['object'].data.root_pos_w[:,2]-env.scene.env_origins[:,2])
        assert float(initial_object_height.max())<.95
        report.update(status='stepping',body_joints=body[1],physical_joint_count=len(robot.joint_names),
            action_dimension=31,actor_observation_dimension=102,hand_open_close_target_deltas=hand_delta,
            reference_mutation_actor_difference=float((before-after).abs().max()),
            waist_limits={n:limits[robot.joint_names.index(n)].tolist()
                          for n in ('waist_roll_joint','waist_pitch_joint')},
            asset_sha256=sha(ASSET),scene_sha256=SCENE_SHA,reference_bank_loaded=env._reference_bank is not None,
            geometry_calibration_max_position_errors_m=geometry_errors,
            geometry_calibration_max_rotation_matrix_errors=geometry_rotation_errors,
            measured_object_initial_height_range=[float(initial_object_height.min()),float(initial_object_height.max())],
            geometry_matches_pinned_urdf=max(geometry_errors)<1e-4 and max(geometry_rotation_errors)<1e-4)
        save()
        start=time.monotonic();resets=0;max_reward=0;max_waist=0
        waist_max=torch.zeros(2,device=env.device)
        for k in range(args.steps):
            action.zero_();action[:,-2:]=1
            # Small independent waist commands verify neither restored joint is locked.
            action[:,BODY_JOINTS.index('waist_roll_joint')]=.08
            action[:,BODY_JOINTS.index('waist_pitch_joint')]=-.08
            obs,reward,terminated,truncated,extras=env.step(action)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            resets+=int((terminated|truncated).sum())
            max_reward=max(max_reward,float(reward.abs().max()))
            waist=robot.data.joint_pos[:,[robot.joint_names.index(n) for n in ('waist_roll_joint','waist_pitch_joint')]]
            max_waist=max(max_waist,float(waist.abs().max()))
            waist_max=torch.maximum(waist_max,waist.abs().amax(dim=0))
        assert bool((waist_max>.01).all())
        report.update(status='smoke_completed',steps=args.steps,elapsed_simulation_s=time.monotonic()-start,
            resets=resets,max_abs_reward=max_reward,max_abs_waist_angle=max_waist,
            max_abs_each_waist_angle=waist_max.tolist(),
            gpu_torch_peak_bytes=torch.cuda.max_memory_allocated(args.device),
            object_initial_height=.90,success_height=.95,success_cannot_be_true_at_reset=True,
            scope='real PhysX scene/control/observation witness; no policy training or tracking-success claim')
    else:
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
        runner_cfg=G1FlatPPORunnerCfg()
        runner_cfg.max_iterations=args.max_iterations
        runner_cfg.experiment_name='dreamcontrol_v2_pick29'
        runner_cfg.seed=20261103
        runner=OnPolicyRunner(RslRlVecEnvWrapper(env),runner_cfg.to_dict(),
                              log_dir=str(args.output_dir/'ppo'),device=args.device)
        from stage3_resume_contract import resume_counts,verify_restoration
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        source=saved['infos']['provenance']
        assert source['asset_sha256']==sha(ASSET) and source['scene_sha256']==SCENE_SHA
        assert source['reference_bank_sha256']==env._reference_bank.manifest_sha256
        assert source['policy_actions']==31 and source['policy_observations']==102 and not source['actor_reference_input']
        completed,remaining=resume_counts(saved['iter'],args.max_iterations)
        parent_sha=sha(args.resume)
        runner.load(str(args.resume),load_optimizer=True)
        restoration=verify_restoration(runner,saved)
        # Stock load() restores the completed index; advance once so it is not trained twice.
        runner.current_learning_iteration=completed
        if hasattr(runner.alg,'learning_rate'):
            runner.alg.learning_rate=float(runner.alg.optimizer.param_groups[0]['lr'])
        resume_receipt={'checkpoint':str(args.resume),'checkpoint_sha256':parent_sha,
            'saved_iteration_index':int(saved['iter']),'completed_updates':completed,
            'remaining_updates':remaining,'new_total_updates':args.max_iterations,
            'exact_restoration':restoration,'adaptive_learning_rate':float(runner.alg.optimizer.param_groups[0]['lr']),
            'simulator_state_reinitialized':True,'bit_identical_simulation_continuation_claimed':False}
        (args.output_dir/'resume_receipt.json').write_text(json.dumps(resume_receipt,indent=2))
        assert sha(args.resume)==parent_sha
        provenance={'asset_sha256':sha(ASSET),'scene_sha256':SCENE_SHA,
            'reference_bank_sha256':env._reference_bank.manifest_sha256,
            'diffusion_checkpoint_sha256':prior_status['last_checkpoint']['sha256'],
            'body_joint_names':BODY_JOINTS,'policy_actions':31,'policy_observations':102,
            'actor_reference_input':False,'num_envs':args.num_envs,'max_iterations':args.max_iterations,
            'resume':resume_receipt,
            'runner_config':runner_cfg.to_dict(),
            'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('stage3_pick29_*.py')}}
        (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2))
        save_original=runner.save
        def save_bound(path,infos=None):
            if Path(path).exists():raise RuntimeError('checkpoint overwrite refused: '+str(path))
            return save_original(path,infos={'provenance':provenance,'user_infos':infos})
        runner.save=save_bound
        log_original=runner.log
        def log_progress(locs,width=80,pad=35):
            log_original(locs,width=width,pad=pad)
            metrics={'iteration_index':int(locs['it']),'policy_updates':int(locs['it'])+1,
                     'collection_time_s':float(locs['collection_time']),'learn_time_s':float(locs['learn_time'])}
            for key,value in locs.get('loss_dict',{}).items():
                metrics[key]=float(value)
            if not all(torch.isfinite(p).all() for p in runner.alg.policy.parameters()):
                raise RuntimeError('nonfinite PPO policy parameters')
            with (args.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            report.update(policy_updates=metrics['policy_updates'],last_metrics=metrics)
            save()
            if metrics['policy_updates']==completed+1:
                runner.save(str(args.output_dir/'ppo/resume_first_update.pt'),infos={'purpose':'first actual update after budget change'})
        runner.log=log_progress
        step_original=env.step
        def guarded_step(action):
            if env.common_step_counter%100==0 and shutil.disk_usage('/data').free<8*1024**3+356*1024**2:
                raise RuntimeError('disk reserve reached during PPO; preserve checkpoints and stop this job')
            return step_original(action)
        env.step=guarded_step
        report.update(status='resume_verified' if args.verify_resume_only else 'ppo_running',
                      max_iterations=args.max_iterations,policy_updates=completed,resume=resume_receipt,
                      reference_bank_sha256=env._reference_bank.manifest_sha256)
        save()
        if not args.verify_resume_only:
            (args.output_dir/'ppo').mkdir(exist_ok=False)
            # RSL initializes this in learn(), while our verified snapshot precedes it.
            runner.logger_type=runner.cfg.get('logger','tensorboard').lower()
            assert runner.logger_type=='tensorboard', 'pre-learn snapshot requires the local logger'
            next_iteration=runner.current_learning_iteration
            runner.current_learning_iteration=completed-1
            runner.save(str(args.output_dir/'ppo/resume_snapshot.pt'),infos={'purpose':'state immediately after verified restoration'})
            runner.current_learning_iteration=next_iteration
            runner.learn(num_learning_iterations=remaining,init_at_random_ep_len=False)
            report.update(status='ppo_completed',policy_updates=args.max_iterations)
    report['total_elapsed_s']=time.monotonic()-began
    save();print('PICK29_RESULT '+json.dumps(report),flush=True)
except BaseException as e:
    report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    save();print(report['traceback'],flush=True)
    raise
finally:
    if env is not None:env.close()
    if app is not None:app.close(wait_for_replicator=False)
