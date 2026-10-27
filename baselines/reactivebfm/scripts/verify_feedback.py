"""Posthoc simulation-feedback consistency check, not a motion-quality metric.

Independently interpolates recorded *actual* simulator states without importing
the planner or its resampling code. Intended for constant A-pose initialization
and the wrapper's fixed application window. No dataset ground truth is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run):
    run = Path(run)
    receipt_path = run / 'receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['status'] == 'PHYSICAL_CLOSED_LOOP_EXECUTED'
    assert receipt['reference_forcing'] is False and receipt['replan_prefix_mode'] == 'real'
    assert receipt['feedback_timebase'].startswith('MuJoCo simulation time')
    files = list((run/'session').rglob('motion_buffers.npz'))
    assert len(files) == 1
    prefix_path = run / 'planner_prefixes.npz'
    with np.load(prefix_path, allow_pickle=False) as p:
        prefix = p['prefix']; anchors = p['anchor_sim_seconds']; fps = float(p['fps'])
    with np.load(files[0], allow_pickle=False) as p:
        states = p['real_robot_motion36']; steps = p['real_robot_policy_step']
    assert prefix.ndim == 3 and prefix.shape[2] == 36 and len(prefix) >= 2
    assert len(states) == receipt['policy_steps'] and np.array_equal(steps, np.arange(len(states)))
    assert np.isfinite(prefix).all() and np.isfinite(states).all()
    dt = receipt['simulation_seconds'] / receipt['policy_steps']
    times = steps * dt
    scalar = np.r_[0:3, 7:36]
    max_linear = 0.; max_quaternion = 0.; max_anchor = 0.
    assert np.max(np.abs(prefix[0] - states[0])) < 1e-5, 'Expected constant A-pose initialization'
    for index, (provided, anchor, call) in enumerate(zip(prefix, anchors, receipt['planner_calls'], strict=True)):
        max_anchor = max(max_anchor, abs(anchor-call['physics_steps_before']*dt))
        targets = anchor - np.arange(prefix.shape[1]-1, -1, -1) / fps
        # Prefix must use observations available at that replan, never future states.
        end = int(call['physics_steps_before']) + 1
        t = times[:end]; observed = states[:end]
        expected = np.stack([np.interp(targets, t, observed[:,j]) for j in scalar], axis=1)
        max_linear = max(max_linear, float(np.max(np.abs(expected-provided[:,scalar]))))
        if len(t)==1:
            q = np.repeat(observed[:1,3:7], len(targets), axis=0)
        else:
            q = Slerp(t, Rotation.from_quat(observed[:,3:7]))(np.clip(targets,t[0],t[-1])).as_quat()
        actual = provided[:,3:7].astype(np.float64)
        actual /= np.linalg.norm(actual,axis=1,keepdims=True)
        error = np.minimum(np.max(np.abs(q-actual),axis=1),np.max(np.abs(q+actual),axis=1))
        max_quaternion = max(max_quaternion, float(error.max()))
    changed = float(np.max(np.abs(prefix[1:,:,scalar]-prefix[0:1,:,scalar])))
    passed = max_linear < 1e-5 and max_quaternion < 1e-5 and max_anchor < 1e-8 and changed > 1e-4
    return {'status':'FEEDBACK_CONSISTENCY_PASSED' if passed else 'FEEDBACK_CONSISTENCY_FAILED',
        'evaluation_type':'simulation_only', 'scope':'Recorded state-to-prefix wiring only; no motion/control quality or paper-performance claim.',
        'planner_calls_checked':len(prefix), 'actual_states_checked':len(states),
        'max_linear_component_error':max_linear, 'max_sign_invariant_quaternion_component_error':max_quaternion,
        'max_anchor_sim_time_error_seconds':max_anchor, 'max_noninitial_prefix_scalar_change':changed,
        'input_sha256':{str(p.relative_to(run)):sha(p) for p in [receipt_path,prefix_path,files[0]]},
        'verifier_sha256':sha(__file__)}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=verify(a.run_dir)
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    raise SystemExit(0 if result['status']=='FEEDBACK_CONSISTENCY_PASSED' else 1)
