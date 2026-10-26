"""One real SDK consuming asynchronous generated frames; records deadlines honestly."""
import argparse,json,os,pathlib,time,traceback,queue,threading
from isaaclab.app import AppLauncher
from online_quick_protocol import ROOT,sha,write,load_spec,request,response
p=argparse.ArgumentParser();p.add_argument('--run',required=True);AppLauncher.add_app_launcher_args(p);a=p.parse_args();s,sh=load_spec()
assert a.headless and a.device=='cuda:1' and set(os.sched_getaffinity(0))=={1,2,3}
out=ROOT/'online_private'/a.run;sdk=out/'sdk';sdk.mkdir(mode=0o700,exist_ok=False)
r={'status':'STARTING','spec_sha256':sh,'optimizer_updates':0,'writers_closed':False,'output_writers_closed':False,'stream_input_kind':'online_generated_reference',
   'semantic_quality_accepted':False,'paper_complete':False,'final_reference_endpoint_verified':False}
app=sim=trace=None;events=[];writer=None;write_queue=None;write_errors=[]
reader=None;reader_stop=threading.Event();read_requests=queue.Queue(maxsize=1);read_results=queue.Queue(maxsize=1);read_errors=[]
def drain_reader():
    global reader
    if reader is not None:
        reader_stop.set();reader.join(timeout=10);assert not reader.is_alive();reader=None
    assert not read_errors,read_errors
def drain_writer():
    global writer
    if writer is not None:
        write_queue.put(None,timeout=10);writer.join(timeout=10)
        assert not writer.is_alive(),'Trace writer did not close'
        writer=None
        assert not write_errors,write_errors
def save():write(sdk/'receipt.json',r)
save()
try:
    import numpy as np
    first=response(out,0);assert first and first['frames']==34
    app=AppLauncher(a).app
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(s['sdk_torch_fraction'],torch.device(a.device))
    r['sdk_torch_cpu_threads']=1
    torch.manual_seed(20261026);torch.cuda.manual_seed_all(20261026)
    from paper_filter_config import make
    from validated_contact_profile import apply as apply_contact_profile
    from apply_tracking_filter_v3 import contact_signature
    from quick_pilot_protocol import load_teacher,authorize_teacher
    from quick_pilot_trace import ExecutionTrace,FIELDS,WIDTH
    from textop_tracker.tasks.tracking.mdp.commands_multi import MotionCommand
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.math import yaw_quat,quat_mul,quat_inv,quat_apply
    from rsl_rl.runners import OnPolicyRunner
    import gymnasium as gym
    warm=json.loads((ROOT/'configs/tracker-env-spec.json').read_text());quick=json.loads((ROOT/'configs/quick-pilot-spec.json').read_text())
    teacher=load_teacher(ROOT,quick);checkpoint=authorize_teacher(quick,teacher);assert sha(checkpoint['path'])==checkpoint['sha256'] and sha(warm['robot_urdf'])==s['urdf_sha256']
    class StreamCommand(MotionCommand):
        def __init__(self,cfg,env):
            super().__init__(cfg,env)
            self.buffer_length=s['buffer_capacity'];self._init_buffers();self.available=0;self.latest=None;self.host_frame=0
            self.motion.file_lengths[0]=s['buffer_capacity']
            self.install(first)
        def install(self,item,arrays=None):
            if arrays is None:
                with np.load(out/item['npz'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files if k not in ['qpos_29','normalized_features','fps']}
            assert item['available_exclusive']==len(arrays['joint_pos'])-1
            if self.latest is not None:
                for k,v in self.latest.items():assert np.array_equal(v[:self.available],arrays[k][:self.available]),'Changed already available reference'
            self.latest=arrays;self.available=item['available_exclusive'];self._update_buffers()
        def _update_buffers(self,env_ids=None):
            if not hasattr(self,'latest') or self.latest is None:return
            n=self.available
            for key,array in self.latest.items():
                target=getattr(self,key+'_buffer');target.fill_(float('nan'))
                if key.startswith('body_'):array=array[:,self.body_indexes.cpu().numpy()]
                target[0,:n]=torch.as_tensor(array[:n],device=self.device,dtype=target.dtype)
                assert torch.isfinite(target[0,:n]).all()
            self.buffer_start_time.zero_()
        def guard(self,extra=0):
            assert self.host_frame+self.cfg.future_steps+extra<=self.available,'Online lookahead underflow; no clamping or last-frame fill permitted'
        def _resample_command(self,env_ids):
            if len(env_ids):self.host_frame=0
            super()._resample_command(env_ids)
        def _update_command(self):
            self.guard(1);self.host_frame+=1;super()._update_command();self.guard()
    # Guard every reference-property access, including constructor/reset observations.
    for name in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w',
                 'anchor_pos_w','anchor_quat_w','anchor_lin_vel_w','anchor_ang_vel_w',
                 'motion_joint_pos','motion_joint_vel','motion_anchor_pos','motion_anchor_quat']:
        original=getattr(MotionCommand,name)
        def guarded(self,prop=original):
            self.guard();return prop.fget(self)
        setattr(StreamCommand,name,property(guarded))
    cfg,agent=make(warm,[str(out/first['npz'])],sdk/'usd',phase='strict',num_envs=1)
    # Explicit alternative for a single live robot: CPU PhysX; neural inference stays GPU1.
    # Full-scale training and every earlier GPU-physics experiment remain unchanged.
    cfg.sim.device='cpu';r['physics_device']='cpu';r['policy_device']=a.device
    plan=json.loads((ROOT/'configs/filter-training-plan.json').read_text());r['contact_profile']=apply_contact_profile(cfg,sdk/'usd',ROOT,plan['contact_profile'])
    assert contact_signature(r['contact_profile'])==contact_signature(teacher['training']['contact_profile'])
    assert cfg.commands.motion.future_steps==5 and cfg.decimation==4 and cfg.terminations.anchor_pos.params['threshold']==.25 and cfg.terminations.ee_body_pos.params['threshold']==.5
    cfg.commands.motion.class_type=StreamCommand;cfg.commands.motion.enable_adaptive_sampling=False;cfg.commands.motion.start_from_zero_step=True
    cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={};cfg.commands.motion.joint_position_range=(0.,0.);cfg.commands.motion.random_static_prob=-1.
    cfg.episode_length_s=s['buffer_capacity']/50;cfg.terminations.time_out=None
    gym_env=gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg);wrapped=RslRlVecEnvWrapper(gym_env,clip_actions=None)
    raw=gym_env.unwrapped;sim=raw.sim;m=raw.command_manager.get_term('motion')
    assert m.robot.joint_names==s['joint_names'] and m.robot.body_names==s['body_names'] and wrapped.get_observations()[0].shape==(1,557)
    runner=OnPolicyRunner(wrapped,agent.to_dict(),log_dir=None,device=a.device);runner.load(checkpoint['path'],load_optimizer=False);base_policy=runner.get_inference_policy(device=a.device)
    def policy(obs):return base_policy(obs.to(a.device))
    def no_optimizer(*args,**kwargs):raise AssertionError('Online validation may not optimize')
    runner.alg.optimizer.step=no_optimizer
    def align():
        n=len(m.cfg.body_names);ap=m.anchor_pos_w[:,None,:].repeat(1,n,1);aq=m.anchor_quat_w[:,None,:].repeat(1,n,1)
        rp=m.robot_anchor_pos_w[:,None,:].repeat(1,n,1).clone();rq=m.robot_anchor_quat_w[:,None,:].repeat(1,n,1);rp[...,2]=ap[...,2];dq=yaw_quat(quat_mul(rq,quat_inv(aq)))
        m.body_pos_relative_w=rp+quat_apply(dq,m.body_pos_w-ap);m.body_quat_relative_w=quat_mul(dq,m.body_quat_w)
    def snap(actions):
        data={'reference_frame':m.time_steps[:,None],'root_pos':m.robot.data.root_pos_w-raw.scene.env_origins,'root_quat_wxyz':m.robot.data.root_quat_w,
            'joint_pos':m.robot_joint_pos,'reference_joint_pos':m.joint_pos,'actions':actions,'body_pos':m.robot_body_pos_w-raw.scene.env_origins[:,None,:],
            'reference_body_pos':m.body_pos_relative_w-raw.scene.env_origins[:,None,:]}
        return {k:v.detach().cpu().numpy()[0].copy() for k,v in data.items()}
    captured={};capture={'active':False};oldcompute=raw.termination_manager.compute
    def before_reset(*args,**kwargs):
        v=oldcompute(*args,**kwargs)
        if capture['active']:
            captured['poses']=snap(capture['actions']);captured['terms']={n:bool(raw.termination_manager.get_term(n)[0]) for n in raw.termination_manager.active_terms}
        return v
    raw.termination_manager.compute=before_reset
    budget={'used':0,'maximum':64*1024**2};trace=ExecutionTrace(sdk/'execution_private',sh,s['control_steps']+1,m.robot.joint_names,budget=budget)
    trace.meta.update(input_schema='textop.quick_online.v1',reference_frames_meaning='bounded control duration plus initial sample, not complete generated stream');trace._meta()
    gym_env.reset();align();observations=raw.observation_manager.compute()
    # Explicit cold-start JIT/physics warm-up; all substeps counted separately, then reset.
    warmup=[]
    with torch.inference_mode():
        for k in range(s['warmup_control_steps']):
            t=time.monotonic();act=policy(observations['policy']);observations,rew,ter,trun,extra=gym_env.step(act);torch.cuda.synchronize(a.device)
            warmup.append({'step':k,'seconds':time.monotonic()-t,'terminated':bool(ter[0]),'truncated':bool(trun[0])})
    write(sdk/'warmup.private.json',warmup);r.update(warmup_control_calls=len(warmup),warmup_physics_substeps=4*len(warmup))
    with torch.inference_mode():
        gym_env.reset();align();observations=raw.observation_manager.compute();trace.append(snap(torch.zeros((1,29),device=a.device)))
    # Warm deterministic inference only, no physics or reset advances, before starting wall-clock deadlines.
    with torch.inference_mode():policy(observations['policy']);torch.cuda.synchronize(a.device)
    # Live inference never consumes a reward or critic. Verify identical privileged
    # observations before reusing policy input; skip reward *evaluation*, not weights
    # or thresholds, only in this inference-only timed phase. Returned zeros are
    # API placeholders and are never reported as measured rewards.
    assert torch.equal(observations['policy'],observations['critic'])
    full_observation_compute=raw.observation_manager.compute
    def inference_observations():
        value=raw.observation_manager.compute_group('policy')
        return {'policy':value,'critic':value}
    raw.observation_manager.compute=inference_observations
    raw.reward_manager.compute=lambda dt:torch.zeros_like(raw.reward_buf)
    r.update(live_reward_evaluation=False,returned_reward_zeros_are_api_placeholders=True,
             identical_privileged_critic_input_reused=True,training_reward_configuration_modified=False)
    # Read-only wall profiling of existing components; no inserted CUDA synchronization.
    component_times={}
    def instrument(obj,name,label):
        original=getattr(obj,name)
        def measured(*args,**kwargs):
            t=time.monotonic()
            try:return original(*args,**kwargs)
            finally:component_times[label]=component_times.get(label,0.)+time.monotonic()-t
        setattr(obj,name,measured)
    for obj,name,label in [(raw.sim,'step','sim_step'),(raw.scene,'update','scene_update'),
                           (raw.reward_manager,'compute','reward_compute'),(raw.command_manager,'compute','command_compute'),
                           (raw.observation_manager,'compute','observation_compute'),(raw.termination_manager,'compute','termination_and_capture'),
                           (raw.action_manager,'apply_action','apply_action'),(raw.scene,'write_data_to_sim','write_data')]:instrument(obj,name,label)
    write_queue=queue.Queue(maxsize=16)
    def write_trace():
        os.sched_setaffinity(0,{0})
        try:
            while True:
                value=write_queue.get()
                if value is None:break
                trace.append(value)
        except BaseException as e:write_errors.append(repr(e))
    writer=threading.Thread(target=write_trace,name='bounded_private_trace_writer');writer.start()
    def read_reference_packets():
        os.sched_setaffinity(0,{0})
        try:
            while not reader_stop.is_set():
                try:index=read_requests.get(timeout=.01)
                except queue.Empty:continue
                item=None
                while not reader_stop.is_set() and item is None:
                    item=response(out,index)
                    if item is None:reader_stop.wait(.001)
                if item is None:break
                with np.load(out/item['npz'],allow_pickle=False) as z:
                    arrays={k:z[k].copy() for k in z.files if k not in ['qpos_29','normalized_features','fps']}
                read_results.put((index,item,arrays),timeout=1)
        except BaseException as e:read_errors.append(repr(e))
    reader=threading.Thread(target=read_reference_packets,name='verified_reference_packet_reader');reader.start()
    pending=None;next_index=1;phase=0;switch=None;consumed_switch=None;received=[first];calls=0;failed=False
    r.update(status='RUNNING_ONLINE',teacher_checkpoint_sha256=checkpoint['sha256'],initial_prefetch_frames=34,
             verified_packet_loading_off_control_thread=True,io_thread_cpu_affinity=[0]);save()
    journal=(sdk/'timing.private.jsonl').open('x')
    # Warm the exact neural and finite-check kernels; no control step or reference advance.
    with torch.inference_mode():
        warm_action=policy(observations['policy']);assert torch.isfinite(warm_action).all()
        assert torch.isfinite(observations['policy']).all();torch.cuda.synchronize(a.device)
    clock=time.monotonic()
    with torch.inference_mode():
        for step in range(s['control_steps']):
            scheduled=clock+step*.02;time.sleep(max(0.,scheduled-time.monotonic()));tick=time.monotonic();component_times.clear()
            if step==s['switch_control_step']:
                phase=1;switch={'step':step,'text':'wave your right hand','submitted_monotonic':time.monotonic()};write(out/'hot_text_command.json',switch)
            assert not read_errors,read_errors
            if pending is not None:
                try:packet=read_results.get_nowait()
                except queue.Empty:packet=None
                if packet is not None:
                    index,item,arrays=packet;assert index==pending
                    m.install(item,arrays);received.append(item);events.append({'event':'sdk_received','index':pending,'at':time.monotonic(),'available_exclusive':m.available})
                    pending=None
            if pending is None and m.available-step<s['prefetch_low_watermark']:
                req=request(out,next_index,phase);pending=next_index;next_index+=1;read_requests.put_nowait(pending)
                events.append({'event':'requested','request':req})
            m.guard(1);assert int(m.time_steps[0])==m.host_frame==step
            # Buffer extensions can change the previously provisional edge only; consumed range is stable.
            poll_done=time.monotonic()
            actions=policy(observations['policy']);assert torch.isfinite(actions).all();capture.update(active=True,actions=actions);policy_done=time.monotonic()
            observations,rewards,terminated,truncated,extras=gym_env.step(actions);torch.cuda.synchronize(a.device);capture['active']=False
            physics_done=time.monotonic();assert not write_errors,write_errors;write_queue.put_nowait(captured['poses']);calls=step+1
            assert torch.isfinite(observations['policy']).all() and torch.isfinite(rewards).all()
            if consumed_switch is None:
                switched=[x for x in received if x['phase']==1 and x['first_new_frame']<=step]
                if switched:consumed_switch={'step':step,'at':physics_done,'response_index':switched[0]['index'],'first_new_frame':switched[0]['first_new_frame']}
            end=time.monotonic();row={'step':step,'scheduled_start':scheduled,'actual_start':tick,'poll_done':poll_done,'policy_done':policy_done,'physics_done':physics_done,'tick_done':end,
                'deadline':scheduled+.02,'deadline_miss_seconds':max(0.,end-scheduled-.02),'available_exclusive':m.available,'current_and_future_max':step+5,'pending_index':pending,'component_seconds':dict(component_times)}
            journal.write(json.dumps(row)+'\n');journal.flush();events.append(row)
            if bool(terminated[0]) or bool(truncated[0]):failed=bool(terminated[0]);break
    control_loop_elapsed=time.monotonic()-clock
    journal.close();drain_reader();drain_writer();trace.stream.close()
    rawposes=np.fromfile(trace.path,dtype='<f4').reshape(calls+1,WIDTH);arrays={};offset=0
    for key,shape in FIELDS:
        width=int(np.prod(shape));arrays[key]=rawposes[:,offset:offset+width].reshape((calls+1,)+shape);offset+=width
    arrays.update(time_seconds=np.arange(calls+1)/50.,joint_names=np.array(m.robot.joint_names),runtime_kind=np.array('isaaclab_physics'),capture_phase=np.array(trace.meta['phase']))
    target=trace.folder/'executed_poses.private.npz';np.savez_compressed(target,**arrays)
    trace.meta.update(status='CLOSED_BOUNDED_ONLINE_TRIAL',samples=calls+1,control_steps=calls,physical_failure=failed,binary_sha256=sha(trace.path),npz_sha256=sha(target),whole_motion_completion_claimed=False);trace._meta()
    rows=[x for x in events if 'deadline' in x];miss=[x['deadline_miss_seconds'] for x in rows]
    body=np.linalg.norm(arrays['body_pos'][1:]-arrays['reference_body_pos'][1:],axis=-1)
    write(sdk/'events.private.json',events)
    r.update(status='ONLINE_BOUNDED_TRIAL_CLOSED',wall_clock_start=clock,control_loop_elapsed_seconds=control_loop_elapsed,writers_closed=True,output_writers_closed=True,async_trace_drained=True,actual_control_calls=calls,actual_physics_substeps=calls*4,physical_failure=failed,
        termination_terms=captured['terms'],full_bounded_duration=calls==s['control_steps'],stream_underruns=0,hot_command=switch,switched_target_consumed=consumed_switch,
        deadline_misses=sum(x>0 for x in miss),max_deadline_miss_seconds=max(miss),wall_elapsed_seconds=time.monotonic()-clock,
        actual_50hz_deadlines_passed=all(x==0 for x in miss),source_model_quality_accepted=False,body_current_target_mean_m=float(body.mean()),
        joint_current_target_rmse_rad=float(np.sqrt(np.mean((arrays['joint_pos'][1:]-arrays['reference_joint_pos'][1:])**2))),
        trace_sha256=sha(target),events_sha256=sha(sdk/'events.private.json'),timing_sha256=sha(sdk/'timing.private.jsonl'),peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(a.device))
    save()
except BaseException as e:
    try:drain_reader()
    except BaseException as re:r['packet_reader_error']=repr(re)
    if 'raw' in locals():r['actual_env_substeps_including_warmup']=int(raw._sim_step_counter)
    try:drain_writer()
    except BaseException as we:r['trace_writer_error']=repr(we)
    if trace is not None and not trace.stream.closed and 'capture' in locals() and capture.get('active') and captured.get('poses') is not None:
        if trace.meta['samples']==calls+1:trace.append(captured['poses'])
    if trace is not None and not trace.stream.closed:trace.close_incomplete()
    r.update(status='FAILED_ONLINE_SDK',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
    if app is not None:
        from bounded_headless_close import close
        close(app,r,save,sdk,simulation=sim,successful_statuses=('ONLINE_BOUNDED_TRIAL_CLOSED',))
