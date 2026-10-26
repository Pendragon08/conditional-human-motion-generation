"""CPU post-hoc terminal reference metrics; no simulation or acceptance threshold."""
import numpy as np

def endpoint_metrics(trace,reference,trial,reference_joint_names,reference_body_names):
    frames=int(trial['reference_frames']);steps=int(trial['control_steps'])
    assert frames>=2 and len(trace['time_seconds'])==steps+1<=frames
    assert trace['root_pos'].shape==(steps+1,3) and trace['root_quat_wxyz'].shape==(steps+1,4)
    assert trace['joint_pos'].shape==(steps+1,29)
    assert reference['joint_pos'].shape==(frames,29) and reference['body_pos_w'].shape==(frames,30,3)
    assert reference['body_quat_w'].shape==(frames,30,4)
    assert len(set(reference_joint_names))==29 and len(set(reference_body_names))==30
    names=trace['joint_names'].tolist();assert len(names)==29 and set(names)==set(reference_joint_names)
    order=[reference_joint_names.index(n) for n in names];pelvis=reference_body_names.index('pelvis')
    for value in list(trace.values())+list(reference.values()):
        if hasattr(value,'dtype') and value.dtype.kind in 'fiu':assert np.isfinite(value).all()
    assert np.allclose(trace['time_seconds'],np.arange(steps+1,dtype=np.float64)/50,rtol=0,atol=1e-12)
    consumed=trace['reference_frame'].reshape(-1).tolist()
    assert consumed==[0]+list(range(steps)),'Requires original initial0 then pre-autoreset current-target records'
    initial_root=np.asarray(trace['root_pos'][0],dtype=np.float64)-np.asarray(reference['body_pos_w'][0,pelvis],dtype=np.float64)
    initial_joints=np.asarray(trace['joint_pos'][0],dtype=np.float64)-np.asarray(reference['joint_pos'][0,order],dtype=np.float64)
    result={'reference_frames':frames,'control_steps':steps,'terminal_elapsed_seconds':steps/50.,
      'last_consumed_reference_frame':steps-1,'terminal_reference_frame_requested':frames-1,
      'initial_world_root_offset_m':float(np.linalg.norm(initial_root)),
      'initial_joint_rmse_rad':float(np.sqrt(np.mean(initial_joints**2))),
      'reference_endpoint_independently_compared':False,'natural_duration_complete':bool(trial['complete']),
      'physical_failed':bool(trial['failed']),'tracking_acceptance_changed':False,'threshold_applied':False}
    if not trial['complete']:
        assert trial['failed'];result['not_compared_reason']='Failure ended before nominal duration; no extrapolation to missing terminal pose'
        return result
    assert not trial['failed'] and steps==frames-1
    actual=np.asarray(trace['root_pos'][-1],dtype=np.float64);expected=np.asarray(reference['body_pos_w'][-1,pelvis],dtype=np.float64)
    delta=actual-expected
    joints=np.asarray(trace['joint_pos'][-1],dtype=np.float64)-np.asarray(reference['joint_pos'][-1,order],dtype=np.float64)
    qa=np.asarray(trace['root_quat_wxyz'][-1],dtype=np.float64);qr=np.asarray(reference['body_quat_w'][-1,pelvis],dtype=np.float64)
    assert .99<np.linalg.norm(qa)<1.01 and .99<np.linalg.norm(qr)<1.01
    qa/=np.linalg.norm(qa);qr/=np.linalg.norm(qr)
    angle=2*np.arccos(np.clip(abs(float(qa@qr)),0.,1.))
    result.update(reference_endpoint_independently_compared=True,
       terminal_world_root_position_error_m=float(np.linalg.norm(delta)),
       terminal_world_root_horizontal_error_m=float(np.linalg.norm(delta[:2])),
       terminal_world_root_height_error_abs_m=float(abs(delta[2])),
       terminal_root_orientation_geodesic_rad=float(angle),
       terminal_joint_rmse_rad=float(np.sqrt(np.mean(joints**2))),terminal_joint_abs_max_rad=float(np.max(abs(joints))),
       coordinate_definition='Absolute reference world coordinates after scene-origin subtraction; no per-frame yaw/root realignment or initial-offset correction',
       timing_definition='Actual recorded pose at elapsed(T-1)/50 compared independently to referenceT-1; original controller last consumedT-2; no extra integration')
    return result
