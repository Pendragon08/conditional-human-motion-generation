"""Draft full-coverage 1000-actor replay loop; not an executable entry point.

Caller must authenticate actor, references, environment and request, own the
shared lock and supervise resources. The supplied environment must use the
assigned-reference motion command and the predeclared strict terminations.
"""
import hashlib
import json
import pathlib


def rollout(env, policy, rows, output, device, progress, max_calls, max_trace_bytes, trial_indices):
    import torch
    from isaaclab.utils.math import yaw_quat, quat_mul, quat_inv, quat_apply
    from strict_tracking_coverage import StrictCoverage
    from direct_strict_trace import ExecutionTrace
    from ppo5000_heldout_aggregate import aggregate

    if len(rows) != 1880 or not callable(policy) or not callable(progress):
        raise ValueError('complete candidate set and explicit callbacks required')
    raw = env.unwrapped
    motion = raw.command_manager.get_term('motion')
    count = raw.num_envs
    if count != 16 or not hasattr(motion, 'assigned_indices'):
        raise ValueError('expected assigned replay with 16 environments')
    if motion.motion.file_lengths.tolist() != [r['frames'] for r in rows]:
        raise ValueError('loaded references do not match manifest')
    if len(motion.robot.joint_names) != 29 or len(motion.cfg.body_names) != 14:
        raise ValueError('unexpected robot layout')
    if (not isinstance(trial_indices,list) or not trial_indices or len(set(trial_indices))!=len(trial_indices)
            or any(type(i) is not int or not 0<=i<len(rows) for i in trial_indices)):
        raise ValueError('invalid predeclared trial selection')
    ordered = sorted(trial_indices, key=lambda i: (rows[i]['frames'], i))
    planned_calls = sum(max(rows[i]['frames'] for i in ordered[k:k+count])-1
                        for k in range(0, len(ordered), count))
    if planned_calls > max_calls or sum(rows[i]['frames'] for i in ordered)*36*4 > max_trace_bytes:
        raise ValueError('evaluation budget exceeded')
    output = pathlib.Path(output)
    if not output.is_absolute() or not output.is_dir():
        raise ValueError('caller must create a private output directory')
    journal = output/'trials.private.jsonl'
    budget = {'used': 0, 'maximum': max_trace_bytes}
    all_traces, index, records = [], [], []
    control_calls = 0
    term_counts = {name: 0 for name in raw.termination_manager.active_terms}
    captured = {}
    capture = {'active': False, 'samples': 0}
    original_compute = raw.termination_manager.compute

    def snapshot(actions):
        values = dict(reference_frame=motion.time_steps[:,None],
                      root_pos=motion.robot.data.root_pos_w-raw.scene.env_origins,
                      root_quat_wxyz=motion.robot.data.root_quat_w,
                      joint_pos=motion.robot_joint_pos, reference_joint_pos=motion.joint_pos,
                      actions=actions, body_pos=motion.robot_body_pos_w-raw.scene.env_origins[:,None,:],
                      reference_body_pos=motion.body_pos_relative_w-raw.scene.env_origins[:,None,:])
        return {k:v.detach().cpu().numpy().copy() for k,v in values.items()}

    def capture_before_autoreset(*args, **kwargs):
        result = original_compute(*args, **kwargs)
        if capture['active']:
            captured['terms'] = {name:raw.termination_manager.get_term(name).clone()
                                 for name in raw.termination_manager.active_terms}
            captured['poses'] = snapshot(capture['actions'])
            capture['samples'] += 1
        return result

    raw.termination_manager.compute = capture_before_autoreset
    try:
        with journal.open('x', encoding='utf-8') as stream, torch.inference_mode():
            for start in range(0, len(ordered), count):
                batch = ordered[start:start+count]
                padded = batch + [batch[-1]]*(count-len(batch))
                motion.assigned_indices.copy_(torch.tensor(padded, device=device))
                env.reset()
                before = motion.time_steps.clone()
                n = len(motion.cfg.body_names)
                ap = motion.anchor_pos_w[:,None,:].repeat(1,n,1)
                aq = motion.anchor_quat_w[:,None,:].repeat(1,n,1)
                rp = motion.robot_anchor_pos_w[:,None,:].repeat(1,n,1).clone()
                rq = motion.robot_anchor_quat_w[:,None,:].repeat(1,n,1)
                rp[...,2] = ap[...,2]
                delta = yaw_quat(quat_mul(rq,quat_inv(aq)))
                motion.body_pos_relative_w = rp+quat_apply(delta,motion.body_pos_w-ap)
                motion.body_quat_relative_w = quat_mul(delta,motion.body_quat_w)
                if not torch.equal(before,motion.time_steps): raise RuntimeError('alignment changed frame')
                observations = raw.observation_manager.compute()
                initial = snapshot(torch.zeros((count,29),device=device))
                traces, cover = [], []
                for slot,i in enumerate(batch):
                    row=rows[i]
                    key='ldm100k-generated|'+row['segment_id']+'|'+str(row['seed'])+'|'+row['mode']
                    ident=hashlib.sha256(key.encode()).hexdigest()
                    trace=ExecutionTrace(output/'execution_private'/('trial_'+str(start+slot)),
                                         ident,row['frames'],motion.robot.joint_names,budget=budget)
                    all_traces.append(trace);traces.append(trace)
                    trace.append({k:v[slot] for k,v in initial.items()})
                    cover.append(StrictCoverage(row['frames'],0))
                pending=set(range(len(batch)))
                for _ in range(max(rows[i]['frames'] for i in batch)-1):
                    if control_calls >= max_calls: raise RuntimeError('control budget exceeded')
                    actions=policy(observations['policy'])
                    if actions.shape != (count,29) or not torch.isfinite(actions).all():
                        raise ValueError('invalid policy actions')
                    capture['actions']=actions;capture['active']=True;before=capture['samples']
                    observations,rewards,terminated,truncated,_=env.step(actions)
                    capture['active']=False
                    if capture['samples'] != before+1: raise RuntimeError('pre-autoreset capture missing')
                    control_calls+=1
                    for slot in sorted(pending):
                        traces[slot].append({k:v[slot] for k,v in captured['poses'].items()})
                    if not torch.isfinite(observations['policy']).all() or not torch.isfinite(rewards).all():
                        raise ValueError('nonfinite physical output')
                    term,end=terminated.tolist(),truncated.tolist()
                    for slot in sorted(list(pending)):
                        cover[slot].observe(terminated=bool(term[slot]),motion_end=bool(end[slot]))
                        if cover[slot].end_reason=='running':continue
                        for name,flags in captured['terms'].items():term_counts[name]+=int(flags[slot])
                        row=rows[batch[slot]];trial=cover[slot].record()
                        record=dict(segment_id=row['segment_id'],seed=row['seed'],mode=row['mode'],trial=trial)
                        stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                        trace_info=traces[slot].finish(trial);trace_info.update(
                            segment_id=row['segment_id'],seed=row['seed'],mode=row['mode'])
                        records.append(record);index.append(trace_info);pending.remove(slot)
                    progress(len(records),control_calls,budget['used'])
                    if not pending:break
                if pending:raise RuntimeError('unfinished physical trial')
        seeds=sorted({r['seed'] for r in rows})
        if len(ordered)==len(rows):
            summary=aggregate(rows,records,seeds)
        else:
            summary=dict(trials=len(records),complete=sum(r['trial']['complete'] for r in records),
                         failed=sum(r['trial']['failed'] for r in records),
                         incomplete=len(ordered)-len(records),pilot_only=True,
                         selected_indices=trial_indices,selection_used_physical_outcomes=False)
        return dict(summary=summary, traces=index, actual_vectorized_control_calls=control_calls,
                    scene_physics_steps=control_calls*4, private_trace_bytes=budget['used'],
                    termination_counts=term_counts, terminal_poses_captured_before_autoreset=True,
                    physical_tracking_executed=True, online_latency_accepted=False)
    finally:
        raw.termination_manager.compute=original_compute
        for trace in all_traces:trace.close_incomplete()
