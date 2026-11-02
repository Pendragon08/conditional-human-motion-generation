"""Draft supervised mixed-5000 generated-reference physics evaluator.

Do not launch without an independently authenticated request, the dedicated
lock-holding supervisor, complete source bindings and resource supervision.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUNS = {'pilot':'user_augmentation2526_mixed5000_heldout_pilot_v2',
        'full':'user_augmentation2526_mixed5000_heldout_v2'}
SUPERVISOR = ROOT/'scripts/supervise_augmentation2526_mixed5000_heldout_v2.py'
STATUS = 'MIXED5000_GENERATED_HELDOUT_PHYSICS_FINISHED'


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def main():
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--request',required=True);pre.add_argument('--trusted-request-sha256',required=True)
    args,_=pre.parse_known_args()
    request=pathlib.Path(args.request)
    if not __debug__ or os.getuid()!=1013 or request.parent != ROOT/'receipts' or request.resolve()!=request or request.stat().st_uid!=1013:
        raise ValueError('invalid request identity')
    if sha(request)!=args.trusted_request_sha256:raise ValueError('request digest mismatch')
    job=json.loads(request.read_text())
    role=job.get('role');run=RUNS.get(role)
    if run is None or job['run']!=run or job['authorized'] is not True or job['stage']!='heldout' or sha(__file__)!=job['executor_sha256']:
        raise ValueError('wrong evaluation request')
    indices=list(range(0,1880,117))[:16] if role=='pilot' else list(range(1880))
    if job['trial_indices']!=indices:raise ValueError('unexpected evaluation selection')
    cache=ROOT/'step34_20261029'/(run+'_pycache_disabled')
    if cache.exists() or cache.is_symlink() or sys.pycache_prefix!=str(cache) or not sys.dont_write_bytecode:
        raise ValueError('bytecode isolation unavailable')
    sim=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/isaacsim-4.5.0')
    for name,digest in (('python.sh','d4f2275d45e4a367340c32cb2e406cc0753f87f630b15229d1e240669f5c9ebf'),
                        ('setup_python_env.sh','bc1a4cf3e31b6ed579197a1e0a9c2a613c874bb99dd0c15a49173fc42f7459b9')):
        if sha(sim/name)!=digest:raise ValueError('Isaac startup source differs')
    lab=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/IsaacLab/source')
    initial=os.pathsep.join([str(ROOT/'scripts'),str(ROOT/'upstream/TextOpTracker/source/textop_tracker'),
                             str(lab/'isaaclab'),str(lab/'isaaclab_assets'),str(lab/'isaaclab_tasks'),str(lab/'isaaclab_rl')])
    sdk=('kit/python/lib/python3.10','kit/python/lib/python3.10/site-packages','python_packages',
         'exts/isaacsim.simulation_app','extsDeprecated/omni.isaac.kit','kit/kernel/py','kit/plugins/bindings-python',
         'exts/isaacsim.robot_motion.lula/pip_prebundle','exts/isaacsim.asset.exporter.urdf/pip_prebundle',
         'extscache/omni.kit.pip_archive-0.0.0+d02c707b.lx64.cp310/pip_prebundle',
         'exts/omni.isaac.core_archive/pip_prebundle','exts/omni.isaac.ml_archive/pip_prebundle',
         'exts/omni.pip.compute/pip_prebundle','exts/omni.pip.cloud/pip_prebundle')
    expected_path=initial+os.pathsep+str(sim)+'/../../../'+initial+os.pathsep+os.pathsep.join(str(sim/x) for x in sdk)
    if os.environ.get('PYTHONPATH','')!=expected_path:raise ValueError('Isaac module search path differs')
    for name,digest in job['bound_sources'].items():
        p=ROOT/name
        if pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts or p.resolve()!=p or p.stat().st_uid!=1013 or sha(p)!=digest:
            raise ValueError('bound source differs: '+name)
    from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
    pid=os.getppid();found=False
    for _ in range(8):
        if pid<=1:break
        proc=pathlib.Path('/proc')/str(pid)
        argv=(proc/'cmdline').read_bytes().decode().split('\0')
        if proc.stat().st_uid==1013 and str(SUPERVISOR) in argv and str(request) in argv:
            found=True;break
        pid=int((proc/'stat').read_text().rsplit(')',1)[1].split()[1])
    lock=ROOT/'receipts/full_library_capacity.lock';stat=lock.stat()
    if not found or not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(),pid,
                                                  os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):
        raise RuntimeError('supervisor must own the common lock')
    if set(os.sched_getaffinity(0))!={0,1,2,3}:raise ValueError('unexpected CPU affinity')
    from augmentation2526_mixed5000_heldout_inputs_v1 import accept
    accepted=accept(job['input_binding']);rows=accepted['rows']
    parent=ROOT/'step34_20261029';out=parent/run
    out.mkdir(mode=0o700,exist_ok=False)
    rec=dict(status='STARTING_MIXED5000_GENERATED_HELDOUT_PHYSICS',run=run,role=role,
             request_sha256=sha(request),actor_sha256=accepted['actor_sha256'],
             manifest_sha256=accepted['manifest_sha256'],reference_count=len(rows),declared_trials=len(indices),
             physical_tracking_executed=False,optimizer_updates=0,output_writers_closed=False,
             observation_corruption=False,domain_randomization=False,online_latency_accepted=False)
    start=time.monotonic();app=env=None
    def save():rec['elapsed_seconds']=time.monotonic()-start;atomic(out/'receipt.json',rec)
    save()
    try:
        from isaaclab.app import AppLauncher
        parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--trusted-request-sha256',required=True)
        AppLauncher.add_app_launcher_args(parser);a=parser.parse_args()
        if not a.headless or a.device!='cuda:1' or 'CUDA_VISIBLE_DEVICES' in os.environ:
            raise ValueError('unexpected simulator device')
        app=AppLauncher(a).app
        import torch
        import gymnasium as gym
        from textop_tracker.tasks.tracking.mdp.commands_multi import MotionCommand
        from augmentation2526_mixed5000_heldout_actor_v1 import build_actor
        from ppo5000_heldout_env_preparation import make
        from augmentation2526_mixed5000_heldout_rollout_v1 import rollout
        torch.set_num_threads(4);torch.manual_seed(job['seed']);torch.cuda.manual_seed_all(job['seed'])
        torch.cuda.set_per_process_memory_fraction(.65,torch.device(a.device))
        if torch.cuda.mem_get_info(a.device)[0] < job['gpu_memory_floor_bytes']:
            raise RuntimeError('insufficient GPU memory at startup')
        class AssignedReplay(MotionCommand):
            def __init__(self,cfg,env):
                super().__init__(cfg,env)
                self.assigned_indices=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
            def _uniform_sampling(self,env_ids):return self.assigned_indices[env_ids]
        def motion_complete(env,command_name='motion'):
            motion=env.command_manager.get_term(command_name)
            return motion.time_steps >= (motion.motion_length-2).clamp_min(0)
        environment=json.loads((ROOT/'configs/tracker-env-spec.json').read_text())
        if sha(environment['robot_urdf'])!=environment['robot_urdf_sha256']:
            raise ValueError('robot URDF differs')
        cfg=make(job,environment,rows,out/'usd',AssignedReplay,motion_complete)
        actor_path=pathlib.Path(accepted['actor_path'])
        if sha(actor_path)!=accepted['actor_sha256']:raise ValueError('actor changed before load')
        payload=torch.load(actor_path,map_location='cpu',weights_only=False)
        policy=build_actor(payload,accepted['training_binding'],a.device)
        env=gym.make('Tracking-Flat-G1-ProjGravObs-v0',cfg=cfg)
        observations,_=env.reset()
        if tuple(observations['policy'].shape)!=(16,431) or not torch.isfinite(observations['policy']).all():
            raise ValueError('unexpected initial policy observations')
        rec.update(status='RUNNING_MIXED5000_GENERATED_HELDOUT_PHYSICS',physical_tracking_executed=False,
                   num_envs=16,policy_observation_dim=431,action_dim=29);save()
        def progress(completed,calls,trace_bytes):
            rec.update(finished_trials=completed,actual_vectorized_control_calls=calls,private_trace_bytes=trace_bytes,
                       physical_tracking_executed=calls>0)
            if calls%50==0 or completed==len(indices):save()
        result=rollout(env,policy,rows,out,a.device,progress,job['max_control_calls'],job['max_trace_bytes'],indices)
        if sha(actor_path)!=accepted['actor_sha256']:raise ValueError('actor changed during evaluation')
        after=accept(job['input_binding'])
        if after['manifest_sha256']!=accepted['manifest_sha256'] or after['reference_count']!=len(rows):
            raise ValueError('reference inputs changed during evaluation')
        atomic(out/'summary.private.json',result['summary'])
        atomic(out/'executed_traces.private.json',result['traces'])
        rec.update(status=STATUS,output_writers_closed=True,finished_trials=result['summary']['trials'],
                   summary_sha256=sha(out/'summary.private.json'),trace_index_sha256=sha(out/'executed_traces.private.json'),
                   journal_sha256=sha(out/'trials.private.jsonl'),actual_vectorized_control_calls=result['actual_vectorized_control_calls'],
                   scene_physics_steps=result['scene_physics_steps'],private_trace_bytes=result['private_trace_bytes'],
                   termination_counts=result['termination_counts'],terminal_poses_captured_before_autoreset=True,
                   completion_counts=result['summary'],optimizer_updates=0);save()
    except BaseException as error:
        rec.update(status='MIXED5000_HELDOUT_FAILED_PARTIAL_RETAINED',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:
        if app is not None:
            from bounded_headless_close import close
            close(app,rec,save,out,simulation=env.unwrapped.sim if env is not None else None,successful_statuses=(STATUS,))


if __name__=='__main__':main()
