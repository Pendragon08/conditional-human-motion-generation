"""Frozen-policy Isaac Lab feedback interface/continuation pilot; never trains PPO.

All arms use the original reference before the branch. Their actual exposed
prefixes are audited rather than assumed equal. No full PhysX clone is claimed.
References alone change after the branch. Original world goal stays fixed.
"""
import argparse
from pathlib import Path
import sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['capture', 'smoke'], required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--video', action='store_true')
p.add_argument('--steps', type=int, default=500)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
if a.video:
    a.enable_cameras = True
sys.argv = [sys.argv[0]]
launcher = AppLauncher(a)
app = launcher.app

import hashlib
import json
import time
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab.managers import TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.motion_lib.motion_lib_base import MotionLibBase
from isaaclab.utils.math import quat_apply_inverse
from isaac_utils.rotations import slerp

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/data/code/ouyanghao/oyh/humanoid-combat-motion')
ART = ROOT/'artifacts/feedback_world_model_20261006'
POLICY = OLD/'third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/exported/policy.pt'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if a.output.exists():
        raise RuntimeError('Preserve existing output; no duplicate invocation')
    a.output.mkdir(parents=True)
    refs = [46, 47, 48] if a.mode == 'smoke' else list(range(49, 55))
    arms = ['fixed', 'fixed_duplicate', 'predicted', 'actual', 'shuffled'] if a.mode == 'smoke' else ['clean', 'push', 'push_duplicate']
    n = len(refs)*len(arms)
    cfg = parse_env_cfg('Isaac-Motion-Tracking-Kick-v0', device=a.device, num_envs=n)
    cfg.seed = 0
    cfg.observations.policy.enable_corruption = False
    # Single-episode diagnostics: score first fall, never silently reset a robot.
    removed = []
    for name, value in vars(cfg.terminations).items():
        if isinstance(value, TerminationTermCfg):
            setattr(cfg.terminations, name, None)
            removed.append(name)
    cfg.episode_length_s = 20.0
    cfg.viewer.resolution = (640, 480)
    cfg.viewer.eye = (4.0, -4.0, 3.0)
    cfg.viewer.lookat = (1.5, 0.0, .7)
    env = gym.make('Isaac-Motion-Tracking-Kick-v0', cfg=cfg,
                   render_mode='rgb_array' if a.video else None)
    env.reset(seed=0)
    e, robot = env.unwrapped, env.unwrapped.scene['robot']
    lib = e.motion_lib
    original = lib.get_motion_state
    torch.manual_seed(0)
    policy = torch.jit.load(str(POLICY), map_location=a.device).eval()
    joint_ids = [robot.joint_names.index(x) for x in lib.joint_names]
    body_ids = [robot.body_names.index(x) for x in ['pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link']]
    torso = robot.body_names.index('torso_link')
    # Existing exported library order is verified numerically, not inferred
    # from a glob filename order.
    archived = np.load(ROOT/'artifacts/sim2sim_mujoco/interfaces/Kick_references.npz')
    identity_error = np.max(np.abs(lib.dof_pos.detach().cpu().numpy()-archived['dof_pos']))
    assert identity_error < 1e-6
    ids = torch.tensor(refs*len(arms), device=a.device, dtype=torch.long)
    e.motion_ids[:] = ids
    e.start_motion_times[:] = 0
    e.episode_length_buf[:] = 0
    state0 = original(ids, torch.zeros(n, device=a.device))
    q = torch.zeros_like(robot.data.default_joint_pos)
    q[:, joint_ids] = state0['dof_pos']
    robot.write_joint_state_to_sim(q, robot.data.default_joint_vel.clone())
    pos = state0['root_pos']+e.scene.env_origins
    pos[:, 2] += .15  # same official reset height, only before the episode
    robot.write_root_pose_to_sim(torch.cat([pos, state0['root_rot']], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=a.device))
    e.action_manager.reset()
    e.scene.write_data_to_sim()
    e.sim.forward()
    e.scene.update(dt=0.0)
    assert e.step_dt == .02
    context_times = ((np.arange(1, 13)+19)*archived['_motion_lengths'][refs[0]]/
                     (archived['_motion_num_frames'][refs[0]]-1))
    assert np.all(archived['_motion_num_frames'][refs] == archived['_motion_num_frames'][refs[0]])
    assert np.all(archived['_motion_lengths'][refs] == archived['_motion_lengths'][refs[0]])
    assert np.all(archived['_motion_dt'][refs] == archived['_motion_dt'][refs[0]])
    branch_step = int(np.ceil(context_times[-1]/.02))
    branch_time = branch_step*.02
    branch_started = False
    candidates, accepted, candidate_audits = {}, {}, {}
    if a.mode == 'smoke':
        for arm in ['predicted', 'actual', 'shuffled']:
            folder = ROOT/f'third_party/DreamControl/TrajGen/sample_feedback_v3_isaac_20261006_{arm}/Kick_sim2'
            files = list(folder.glob('*_n.pkl'))
            accepted[arm] = [int(x.stem.split('_')[0]) for x in files]
            if files:
                c = MotionLibBase(n, a.device, str(folder))
                c.load_motions()
                mapping = {int(Path(x).stem.split('_')[0]): j for j, x in enumerate(c._motion_data_load)}
                candidates[arm] = (c, mapping)
                assert torch.all(c._motion_num_frames == lib._motion_num_frames[refs[0]])
                assert torch.all(c._motion_lengths == lib._motion_lengths[refs[0]])
                assert torch.all(c._motion_dt == lib._motion_dt[refs[0]])
                limits = robot.data.joint_pos_limits[0, joint_ids]
                excess = torch.maximum(limits[:, 0]-c.dof_pos, c.dof_pos-limits[:, 1]).clamp_min(0)
                candidate_audits[arm] = {
                    'finite': bool(torch.isfinite(c.dof_pos).all() and torch.isfinite(c.transl).all()),
                    'max_hard_joint_limit_excess_rad': float(excess.max()),
                    'samples_with_limit_excess': int((excess > 1e-4).flatten(1).any(-1).sum()),
                    'accepted_count': len(files), 'denominator': 3,
                    'files': [{'source_index': local, 'original_reference': refs[local],
                               'file': str(Path(c._motion_data_load[idx]).relative_to(ROOT)),
                               'sha256': sha(c._motion_data_load[idx])} for local, idx in mapping.items()],
                }
                target_errors = []
                for local, idx in mapping.items():
                    rid = refs[local]
                    event_t = lib.switch_idxs[rid]*lib._motion_dt[rid]
                    v = c.get_motion_state(torch.tensor([idx], device=a.device), event_t.reshape(1))
                    foot = v['global_keypts'][0,13]+lib.offsets[rid]-v['offsets'][0]
                    target_errors.append({'source_index': local,
                                          'reference_virtual_goal_error_at_original_event_m':
                                          float((foot-lib.grab_pos[rid]-lib.offsets[rid]).norm())})
                candidate_audits[arm]['fixed_world_target_reference_errors'] = target_errors

    def switched(motion_ids, times):
        raw = original(motion_ids, times)
        if not branch_started or a.mode != 'smoke':
            return raw
        assert len(motion_ids) == n and torch.equal(motion_ids, ids)
        for k, arm in enumerate(arms):
            if arm not in candidates:
                continue
            c, mapping = candidates[arm]
            samples = [j for j in range(len(refs)) if j in mapping]
            rows = torch.tensor([k*len(refs)+j for j in samples], device=a.device, dtype=torch.long)
            cids = torch.tensor([mapping[j] for j in samples], device=a.device, dtype=torch.long)
            v = c.get_motion_state(cids, times[rows])
            # Suppress candidate-dependent scene placement. Keep original
            # translation, world target, event phase and total episode length.
            shift = raw['offsets'][rows]-v['offsets']
            v['root_pos'] += shift
            v['global_keypts'] += shift[:, None, :]
            blend = ((times[rows]-branch_time)/.2).clamp(0, 1)[:, None]
            for field in ['root_pos', 'dof_pos', 'local_keypts', 'global_keypts']:
                weight = blend[:, :, None] if raw[field].ndim == 3 else blend
                raw[field][rows] = (1-weight)*raw[field][rows]+weight*v[field]
            raw['root_rot'][rows] = slerp(raw['root_rot'][rows], v['root_rot'], blend)
            # grab_pos, offsets and is_closed deliberately untouched.
        return raw

    lib.get_motion_state = switched
    records = {x: [] for x in ['positions', 'quaternions', 'joints', 'velocities', 'actions',
                               'body_xyz', 'reference_body_xyz', 'joint_rmse', 'root_error',
                               'fall', 'goal_error', 'contact_force', 'reference_joint_delta',
                               'active_reference_joint_rmse', 'active_reference_root_error']}
    writer = None
    if a.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(a.output/'actual_isaac_continuation.mp4'), fps=10)
    prefix_errors = []
    prefix_full_state = []
    prev_ref = None
    start = time.time()
    sensor = e.scene['contact_forces']
    contact_idx = [sensor.body_names.index(x) for x in ['left_ankle_roll_link', 'right_ankle_roll_link']]
    goal = state0['grab_pos']+state0['offsets']
    fixed_fields = {x: getattr(lib, x).clone() for x in ['grab_pos', 'offsets', 'switch_idxs']}
    for step in range(a.steps):
        branch_started = step >= branch_step
        assert torch.equal(e.motion_ids, ids) and torch.all(e.episode_length_buf == step)
        assert torch.all(e.start_motion_times == 0)
        times = torch.full((n,), step*.02, device=a.device)
        oref = original(ids, times)
        active_ref = switched(ids, times)
        obs = e.observation_manager.compute()['policy']
        with torch.inference_mode():
            action = policy(obs)
        assert obs.shape == (n, 626) and action.shape == (n, 29)
        assert torch.isfinite(obs).all() and torch.isfinite(action).all()
        qnow = robot.data.joint_pos[:, joint_ids]
        root = robot.data.root_pos_w-e.scene.env_origins
        body = robot.data.body_pos_w[:, body_ids]-e.scene.env_origins[:, None, :]
        reference_body = oref['global_keypts'][:, [0, 7, 13]]
        up = robot.data.projected_gravity_b[:, 2]
        fallen = (root[:, 2] < .3) | (up > -.5)
        values = {
            'positions': root, 'quaternions': robot.data.root_quat_w, 'joints': qnow,
            'velocities': torch.cat([robot.data.root_vel_w, robot.data.joint_vel[:, joint_ids]], dim=-1),
            'actions': action, 'body_xyz': body, 'reference_body_xyz': reference_body,
            'joint_rmse': ((qnow-oref['dof_pos'])**2).mean(-1).sqrt(),
            'root_error': (root-oref['root_pos']).norm(dim=-1),
            'active_reference_joint_rmse': ((qnow-active_ref['dof_pos'])**2).mean(-1).sqrt(),
            'active_reference_root_error': (root-active_ref['root_pos']).norm(dim=-1),
            'fall': fallen, 'goal_error': (body[:, 2]-goal).norm(dim=-1),
            'contact_force': sensor.data.net_forces_w[:, contact_idx].norm(dim=-1),
            'reference_joint_delta': torch.zeros(n, device=a.device) if prev_ref is None else
                                     (active_ref['dof_pos']-prev_ref).abs().max(-1).values,
        }
        prev_ref = active_ref['dof_pos'].clone()
        for key, value in values.items():
            records[key].append(value.detach().cpu().numpy())
        # Duplicate numerical trajectories are compared through the branch.
        if a.mode == 'smoke' and step < branch_step:
            checks = {'root_xyz_m': root, 'root_quaternion': robot.data.root_quat_w,
                      'joint_rad': qnow, 'joint_vel_rad_s': robot.data.joint_vel[:, joint_ids],
                      'root_vel': robot.data.root_vel_w,
                      'last_action': e.action_manager.action, 'current_action': action}
            differences = {key: float(torch.stack([(v[k*len(refs):(k+1)*len(refs)]-v[:len(refs)]).abs().max()
                                                  for k in range(1,len(arms))]).max()) for key,v in checks.items()}
            prefix_full_state.append(differences)
            prefix_errors.append(differences['joint_rad'])
        elif a.mode == 'capture':
            prefix_errors.append(float((qnow[6:12]-qnow[12:18]).abs().max()))
        forces = torch.zeros(n, 1, 3, device=a.device)
        if a.mode == 'capture' and 50 <= step < 60:
            world = torch.zeros(n, 3, device=a.device)
            world[len(refs):, 1] = 80
            forces[:, 0] = quat_apply_inverse(robot.data.body_quat_w[:, torso], world)
        robot.set_external_force_and_torque(forces, torch.zeros_like(forces), body_ids=[torso])
        obs_next, reward, term, trunc, info = env.step(action)
        assert not torch.any(term) and not torch.any(trunc)
        if writer is not None and step % 5 == 0:
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)
        if step % 50 == 0:
            print('ISAAC_FEEDBACK_PHYSICAL_STEP', step, 'branch', branch_step, flush=True)
    if writer is not None:
        writer.close()
    for name, value in fixed_fields.items():
        assert torch.equal(getattr(lib, name), value)
    arrays = {k: np.stack(v) for k, v in records.items()}
    np.savez_compressed(a.output/'physical_rollout.npz', **arrays)
    rows = []
    for k, arm in enumerate(arms):
        for j, ref in enumerate(refs):
            idx = k*len(refs)+j
            available = arm not in ['predicted', 'actual', 'shuffled'] or j in accepted.get(arm, [])
            falls = np.where(arrays['fall'][:, idx])[0]
            first_fall = int(falls[0]) if len(falls) else None
            metric = arrays['joint_rmse'][:, idx].copy()
            if first_fall is not None:
                metric[first_fall:] = 2.0
            # Event window is inherited from original motion library.
            event_s = float(lib.switch_idxs[ref]*lib._motion_dt[ref])
            mask = np.abs(np.arange(a.steps)*.02-event_s) <= .1
            error = arrays['goal_error'][mask, idx]
            rows.append({'arm': arm, 'reference': ref,
                         'retarget_available': available,
                         'physical_policy_trial_eligible': available,
                         'regenerated_physical_trial_executed': available and arm in ['predicted','actual','shuffled'],
                         'missing_candidate_behavior': None if available else 'original fallback for batch safety, excluded from physical benefit, counted as failure',
                         'first_fall_step': first_fall if available else None,
                         'tracking_auc_after_branch_rad_s': float(metric[branch_step:branch_step+90].sum()*.02) if available else None,
                         'is_disturbance_recovery_measure': a.mode == 'capture',
                         'active_reference_joint_rmse_mean_rad': float(arrays['active_reference_joint_rmse'][branch_step:,idx].mean()) if available else None,
                         'goal_event_s': event_s,
                         'virtual_goal_min_error_m': float(error.min()) if len(error) and available else None,
                         'virtual_goal_success': bool(np.any(error < .1)) if len(error) and available else False,
                         'reference_switch_delta_p95_rad': float(np.percentile(arrays['reference_joint_delta'][branch_step:branch_step+20,idx],95)),
                         'physical_entity_contact_evaluated': False})
    result = {'status': 'COMPLETE', 'mode': a.mode, 'platform': 'Isaac Lab / Isaac Sim4.5 PhysX',
              'frozen_policy_sha256': sha(POLICY), 'seed': 0, 'references': refs, 'arms': arms,
              'steps': a.steps, 'step_dt_s': .02, 'branch_step': branch_step, 'blend_s': .2,
              'wall_s': time.time()-start, 'num_environments': n, 'observation_dim': 626,
              'action_dim': 29, 'ppo_updates': 0, 'extra_ik': False,
              'auto_resets_disabled': removed, 'robot_state_reset_at_branch': False,
              'original_goal_offset_and_event_preserved': True,
              'original_library_identity_max_error': float(identity_error),
              'duplicate_joint_difference_max_rad': max(prefix_errors),
              'all_arm_exposed_prefix_max_abs_difference': {key:max(x[key] for x in prefix_full_state) for key in prefix_full_state[0]} if prefix_full_state else None,
              'physical_prefix_equality_certified': False,
              'full_physx_contact_solver_clone_verified': False,
              'accepted_generation_ids': accepted,
              'candidate_audits': candidate_audits,
              'smoke_feedback_source': 'previous nominal MuJoCo sparse residuals' if a.mode=='smoke' else 'new post-push actual Isaac Lab states',
              'formal_feedback_control_gate_evaluated': False,
              'world_force_N': [0,80,0] if a.mode=='capture' else [0,0,0],
              'force_control_steps': [50,60] if a.mode=='capture' else None,
              'context_human_frames': list(range(1,13)), 'context_sample_steps': np.rint(context_times/.02).astype(int).tolist(),
              'rows': rows, 'physical_rollout_sha256': sha(a.output/'physical_rollout.npz')}
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('ISAAC_FEEDBACK_CONTINUATION_COMPLETE', json.dumps(result), flush=True)
    env.close()

try:
    main()
finally:
    app.close()
