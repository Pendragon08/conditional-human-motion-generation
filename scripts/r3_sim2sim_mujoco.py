"""Frozen whole-body DreamControl policy transfer to native MuJoCo.

Explicit torque PD follows the official Sim2Real deployment convention. The
gains, default positions and limits come from the actual Isaac runtime. Native
MuJoCo integrates CPU physics; Torch inference and EGL rendering use the GPU.
This is a zero-shot transfer experiment, not additional RL or motion playback.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import imageio.v2 as imageio
import imageio as imageio_package
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DC = ROOT / 'third_party/DreamControl'
INTERFACE = ROOT / 'artifacts/sim2sim_mujoco/interfaces'
JOINTS = [f'{side}_{j}_joint' for side in ['left', 'right']
          for j in ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']]
JOINTS += ['waist_yaw_joint']
JOINTS += [f'{side}_{j}_joint' for side in ['left', 'right']
           for j in ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def policy_path(task):
    root = Path('/data/code/ouyanghao/oyh/humanoid-combat-motion') if task == 'Kick' else ROOT
    run = '2026-09-27_03-19-31-Kick-a' if task == 'Kick' else '2026-09-29_12-37-46-Punch-a'
    return root / 'third_party/DreamControl/Training/logs/rsl_rl/g1' / run / 'exported/policy.pt'


def rotation(q):
    w, x, y, z = q
    return np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                     [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                     [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])


def multiply(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


class Reference:
    def __init__(self, task):
        self.a = dict(np.load(INTERFACE / f'{task}_references.npz'))
        source = DC / 'Training/isaac_utils/isaac_utils/rotations.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'slerp')
        node.decorator_list = []
        ns = {'torch': torch, 'Tensor': torch.Tensor}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), 'exec'), ns)
        self.slerp = ns['slerp']

    def state(self, i, t):
        a = self.a
        n, length, dt = int(a['_motion_num_frames'][i]), float(a['_motion_lengths'][i]), float(a['_motion_dt'][i])
        # Preserve the official nonstandard frame-blend implementation exactly.
        f0 = int(np.clip(t / length, 0, 1) * (n - 1))
        f1 = min(f0 + 1, n - 1)
        blend = np.clip((max(t, 0) - f0 * dt) / dt, 0, 1)
        def lerp(name):
            return (1 - blend) * a[name][i, f0] + blend * a[name][i, f1]
        q0, q1 = a['quats'][i, f0].copy(), a['quats'][i, f1].copy()
        # Execute the official numerical function, avoiding a NumPy reduction
        # roundoff changing its tiny-angle branch. No quaternion normalization
        # or alternative interpolation is inserted.
        q = self.slerp(torch.from_numpy(q0).unsqueeze(0), torch.from_numpy(q1).unsqueeze(0),
                       torch.tensor([[blend]], dtype=torch.float32))[0].numpy()
        return {'pos': lerp('transl') + a['offsets'][i], 'quat': q,
                'joints': lerp('dof_pos'), 'keypts': lerp('global_keypts') + a['offsets'][i],
                'goal': a['grab_pos'][i] + a['offsets'][i],
                'closed': f0 > float(a['switch_idxs'][i])}

    def target(self, i, t, pos, quat):
        s = self.state(i, t)
        q = quat.copy()
        if q[0] < 0:
            q = -q
        inv = q * np.array([1., -1., -1., -1.])
        return np.concatenate([s['joints'], np.zeros(27), rotation(q).T @ (s['pos'] - pos),
                               multiply(inv, s['quat']), ((s['keypts'] - pos) @ rotation(quat)).ravel()])


def observation(ref, i, t, pos, quat, lin_b, ang_b, joints, velocities, last, default):
    return np.concatenate([lin_b, ang_b, rotation(quat).T @ [0., 0., -1.],
                           joints - default, velocities, last,
                           ref.target(i, t, pos, quat), ref.target(i, t+.1, pos, quat),
                           ref.target(i, t+.2, pos, quat)]).astype(np.float32)


class Transfer:
    def __init__(self, task, device='cuda:0'):
        self.task, self.device = task, device
        self.meta = json.loads((INTERFACE / f'{task}_interface.json').read_text())
        assert self.meta['reference_joint_names'] == JOINTS
        assert self.meta['num_actions'] == 29
        self.ref = Reference(task)
        self.policy_path = policy_path(task)
        assert sha(self.policy_path) == self.meta['policy_sha256']
        self.policy = torch.jit.load(str(self.policy_path), map_location=device).eval()
        checkpoint_path = self.policy_path.parent.parent / 'model_1999.pt'
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        state = self.policy.state_dict()
        actor_equal = all(torch.equal(v.cpu(), checkpoint['model_state_dict'][k])
                          for k, v in state.items() if k.startswith('actor.'))
        norm_equal = all(torch.equal(v.cpu(), checkpoint['obs_norm_state_dict'][k[len('normalizer.'):]])
                         for k, v in state.items() if k.startswith('normalizer.'))
        assert actor_equal and norm_equal
        self.provenance = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha(checkpoint_path),
                           'actor_tensor_equal_to_checkpoint': actor_equal,
                           'normalizer_tensor_equal_to_checkpoint': norm_equal,
                           'adapter_sha256': sha(__file__),
                           'official_slerp_source_sha256': sha(DC / 'Training/isaac_utils/isaac_utils/rotations.py')}
        del checkpoint
        self.scene_path = DC / 'Sim2Real/resources/robots/g1_description/scene_full.xml'
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = .005
        self.model.opt.gravity[:] = [0, 0, -9.81]
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # Disable robot-robot collision as in the training articulation. Floor
        # has bit 2, robot colliders bit 1, so only floor-robot pairs collide.
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == 0:
                self.model.geom_contype[g], self.model.geom_conaffinity[g] = 2, 1
            elif self.model.geom_contype[g] or self.model.geom_conaffinity[g]:
                self.model.geom_contype[g], self.model.geom_conaffinity[g] = 1, 2
        self.model.geom_friction[:, 0] = 1.
        self.data = mujoco.MjData(self.model)
        self.robot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.endpoint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         'right_ankle_roll_link' if task == 'Kick' else 'right_wrist_yaw_link')
        self.defaults = dict(zip(self.meta['joint_names'], self.meta['default_joint_pos']))
        self.default = np.array([self.defaults[j] for j in JOINTS])
        self.ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS])
        assert min(self.ids) >= 0
        self.qadr = self.model.jnt_qposadr[self.ids]
        self.vadr = self.model.jnt_dofadr[self.ids]
        self.kp, self.kd, self.limit = np.zeros(self.model.nu), np.zeros(self.model.nu), np.zeros(self.model.nu)
        self.targets = np.zeros(self.model.nu)
        self.all_joint_names = []
        gains = {}
        for a in self.meta['actuators'].values():
            for idx, name in enumerate(a['joint_names']):
                gains[name] = tuple(a[k][idx] for k in ['stiffness', 'damping', 'effort_limit', 'armature'])
        for a in range(self.model.nu):
            j = int(self.model.actuator_trnid[a, 0])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.all_joint_names.append(name)
            self.kp[a], self.kd[a], self.limit[a], armature = gains[name]
            self.model.dof_armature[self.model.jnt_dofadr[j]] = armature
            self.model.actuator_ctrllimited[a] = True
            self.model.actuator_ctrlrange[a] = [-self.limit[a], self.limit[a]]
            # A motor rather than a position actuator receives explicit PD torque.
            assert self.model.actuator_gainprm[a, 0] == 1 and np.all(self.model.actuator_biasprm[a] == 0)
        self.all_qadr = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        self.all_vadr = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.body_actuators = np.array([self.all_joint_names.index(j) for j in JOINTS])
        self.hand_targets = self.targets.copy()
        for a, name in enumerate(self.all_joint_names):
            if 'hand' not in name:
                continue
            sign = -1 if name.startswith('left') else 1
            if 'thumb_0' in name:
                value = 0.
            elif 'thumb_1' in name:
                value = -sign*np.pi/3
            elif 'thumb_2' in name:
                value = -sign*np.pi/2
            else:
                value = sign*np.pi/2
            self.hand_targets[a] = value

    def reset(self, i):
        mujoco.mj_resetData(self.model, self.data)
        s = self.ref.state(i, 0.)
        self.data.qpos[:3] = s['pos'] + [0, 0, .15]
        self.data.qpos[3:7] = s['quat']
        self.data.qpos[self.qadr] = s['joints']
        self.data.qvel[:] = 0
        self.last = np.zeros(29, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)

    def obs(self, i, t):
        q = self.data.qpos[3:7].copy()
        # Free-joint translational velocity is world-frame; angular is local.
        return observation(self.ref, i, t, self.data.qpos[:3], q,
                           rotation(q).T @ self.data.qvel[:3], self.data.qvel[3:6],
                           self.data.qpos[self.qadr], self.data.qvel[self.vadr], self.last, self.default)

    def action(self, obs):
        with torch.inference_mode():
            return self.policy(torch.from_numpy(obs).unsqueeze(0).to(self.device))[0].cpu().numpy()

    def advance(self, action):
        self.targets[:] = self.hand_targets
        self.targets[self.body_actuators] = .5 * action[:27] + self.default
        max_excess = 0.
        for _ in range(4):
            torque = self.kp * (self.targets - self.data.qpos[self.all_qadr]) - self.kd * self.data.qvel[self.all_vadr]
            max_excess = max(max_excess, float(np.max(np.maximum(np.abs(torque)-self.limit, 0))))
            self.data.ctrl[:] = np.clip(torque, -self.limit, self.limit)
            mujoco.mj_step(self.model, self.data)
        self.last = action.copy()
        mujoco.mj_forward(self.model, self.data)
        return max_excess

    def point(self):
        pos = self.data.xpos[self.endpoint].copy()
        if self.task == 'Punch':
            pos += self.data.xmat[self.endpoint].reshape(3, 3)[:, 0] * .14
        return pos

    def fallen(self):
        return bool(self.data.qpos[2] <= .3 or rotation(self.data.qpos[3:7])[2, 2] <= .5)

    def verify(self):
        rows = []
        for n in range(4):
            s = dict(np.load(INTERFACE / f'{self.task}_snapshot_{n}.npz'))
            lookup = {j: i for i, j in enumerate(self.meta['joint_names'])}
            jids = [lookup[j] for j in JOINTS]
            obs = observation(self.ref, int(s['motion_id']), float(s['motion_time']), s['root_pos'], s['root_quat'],
                              s['root_lin_vel_b'], s['root_ang_vel_b'], s['joint_pos'][jids],
                              s['joint_vel'][jids], s['last_action'], self.default)
            action = self.action(obs)
            # Also inject the Isaac state into MuJoCo coordinates, exercising
            # its qpos/qvel mapping rather than only the algebraic inputs.
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:3], self.data.qpos[3:7] = s['root_pos'], s['root_quat']
            for n_joint, name in enumerate(self.meta['joint_names']):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.data.qpos[self.model.jnt_qposadr[jid]] = s['joint_pos'][n_joint]
                self.data.qvel[self.model.jnt_dofadr[jid]] = s['joint_vel'][n_joint]
            self.data.qvel[:3] = rotation(s['root_quat']) @ s['root_lin_vel_b']
            self.data.qvel[3:6] = s['root_ang_vel_b']
            self.last = s['last_action'].copy()
            mujoco.mj_forward(self.model, self.data)
            mapped = self.obs(int(s['motion_id']), float(s['motion_time']))
            rows.append({'snapshot': n, 'obs_max_abs_error': float(np.max(np.abs(obs-s['obs']))),
                         'action_max_abs_error': float(np.max(np.abs(action-s['action']))),
                         'mujoco_state_mapping_obs_max_abs_error': float(np.max(np.abs(mapped-s['obs'])))})
        passed = all(r['obs_max_abs_error'] < 2e-4 and r['action_max_abs_error'] < 2e-4
                     and r['mujoco_state_mapping_obs_max_abs_error'] < 2e-4 for r in rows)
        return {'task': self.task, 'passed': passed, 'rows': rows, 'obs_dim': 626, 'action_dim': 29}


def render_frame(renderer, transfer, goal):
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [transfer.data.qpos[0], transfer.data.qpos[1], .75]
    camera.distance, camera.azimuth, camera.elevation = 3., 135., -15.
    renderer.update_scene(transfer.data, camera=camera)
    scene = renderer.scene
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        [.04, 0, 0], goal, np.eye(3).ravel(), [1, .3, .1, 1])
    scene.ngeom += 1
    return renderer.render()


class VideoWriter:
    """Use the existing system FFmpeg; no Python plugin/package installation."""
    def __init__(self, path):
        self.path = Path(path)
        self.process = subprocess.Popen(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '640x480', '-r', '25', '-i', '-',
            '-an', '-c:v', 'libx264', '-crf', '22', '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)

    def append_data(self, frame):
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.process.stdin.close()
        rc = self.process.wait()
        if rc != 0 or not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f'Video encode failed: {self.path}, exit={rc}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['Kick', 'Punch'], default='Kick')
    p.add_argument('--mode', choices=['witness', 'verify', 'eval'], default='witness')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-refs', type=int, default=0)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--videos', type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    t = Transfer(args.task, args.device)
    verify = t.verify()
    result_dir = ROOT / 'results/sim2sim_mujoco'
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / f'{args.task}_interface_parity.json').write_text(json.dumps(verify, indent=2))
    if not verify['passed']:
        raise RuntimeError(f'Isaac observation/action parity failed: {verify}')
    print('INTERFACE_PARITY_PASS', json.dumps(verify), flush=True)
    if args.mode == 'verify':
        return
    renderer = mujoco.Renderer(t.model, height=480, width=640)
    from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR
    gl_info = {'renderer': str(glGetString(GL_RENDERER)), 'vendor': str(glGetString(GL_VENDOR))}
    if 'NVIDIA' not in gl_info['vendor']:
        raise RuntimeError(f'GPU EGL witness failed: {gl_info}')
    t.reset(0)
    before = t.data.qpos.copy()
    action = t.action(t.obs(0, 0.))
    for _ in range(10):
        t.advance(action)
    frame = render_frame(renderer, t, t.ref.state(0, 0)['goal'])
    witness = {'task': args.task, 'python': platform.python_version(), 'mujoco': mujoco.__version__,
               'torch': torch.__version__, 'numpy': np.__version__, 'imageio': imageio_package.__version__,
               'device': torch.cuda.get_device_name(torch.device(args.device)), 'gl': gl_info,
               'policy_output_shape': list(action.shape), 'physics_time': t.data.time,
               'state_changed': bool(np.max(np.abs(t.data.qpos-before)) > 1e-6),
               'finite': bool(np.isfinite(t.data.qpos).all()), 'render_shape': list(frame.shape),
               'render_std': float(frame.std()), 'policy_sha256': sha(t.policy_path),
               'scene_sha256': sha(t.scene_path), 'provenance': t.provenance}
    assert witness['state_changed'] and witness['finite'] and witness['render_std'] > 10
    (result_dir / f'{args.task}_environment_witness.json').write_text(json.dumps(witness, indent=2))
    print('SIM2SIM_ENV_WITNESS', json.dumps(witness), flush=True)
    if args.mode == 'witness':
        renderer.close()
        return
    count = len(t.ref.a['_motion_lengths'])
    if args.max_refs:
        count = min(count, args.max_refs)
    records = []
    start = time.monotonic()
    media = ROOT / 'media/sim2sim_mujoco'
    media.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        t.reset(i)
        writer = VideoWriter(media / f'{args.task}_reference_{i:03d}.mp4') if i < args.videos else None
        success, fall, steps, max_excess = False, False, 0, 0.
        errors, joint_errors = [], []
        window_errors = []
        for k in range(args.steps):
            current = k * .02
            action = t.action(t.obs(i, current))
            if not np.isfinite(action).all():
                raise RuntimeError('Nonfinite action')
            max_excess = max(max_excess, t.advance(action))
            now = (k+1) * .02
            ref = t.ref.state(i, now)
            error = float(np.linalg.norm(t.point()-ref['goal']))
            errors.append(error)
            joint_errors.append(float(np.mean((t.data.qpos[t.qadr]-ref['joints'])**2)))
            window = t.ref.state(i, now+.1)['closed'] and not t.ref.state(i, now-.1)['closed']
            if window:
                window_errors.append(error)
                if error < (.1 if args.task == 'Kick' else .05):
                    success = True
            steps = k+1
            if writer is not None and k % 2 == 0:
                writer.append_data(render_frame(renderer, t, ref['goal']))
            if t.fallen():
                fall = True
                break
            if not np.isfinite(t.data.qpos).all():
                raise RuntimeError('Nonfinite simulation state')
        if writer is not None:
            writer.close()
        records.append({'reference_index': i, 'task_success_virtual_target': success,
                        'fallen': fall, 'executed_steps': steps, 'survival_s': steps*.02,
                        'min_goal_distance_m': min(errors),
                        'event_window_min_distance_m': min(window_errors) if window_errors else None,
                        'joint_tracking_rmse_rad': float(np.sqrt(np.mean(joint_errors))),
                        'max_unclipped_pd_torque_excess_nm': max_excess,
                        'video': str(media / f'{args.task}_reference_{i:03d}.mp4') if i < args.videos else None})
        print('TRIAL', args.task, i, success, fall, steps, flush=True)
    summary = {'task': args.task, 'status': 'COMPLETE', 'protocol': 'one deterministic start-at-zero episode per reference; no autoreset; 500 control frames maximum',
               'adapter_revision': 'exact_official_slerp_v2', 'provenance': t.provenance,
               'policy_frozen': True, 'learning': False, 'extra_ik': False,
               'num_references': count, 'policy_sha256': sha(t.policy_path), 'reference_npz_sha256': sha(INTERFACE / f'{args.task}_references.npz'),
               'success_count': sum(r['task_success_virtual_target'] for r in records),
               'success_rate': sum(r['task_success_virtual_target'] for r in records)/count,
               'fall_count': sum(r['fallen'] for r in records),
               'survive_full_horizon_count': sum(r['executed_steps'] == args.steps and not r['fallen'] for r in records),
               'success_and_survive_full_horizon_count': sum(r['task_success_virtual_target'] and r['executed_steps'] == args.steps and not r['fallen'] for r in records),
               'elapsed_wall_s': time.monotonic()-start, 'physics_dt_s': .005, 'policy_dt_s': .02,
               'controller': 'explicit torque PD at 200Hz, training runtime gains/defaults/effort limits, clamp torque',
               'model': 'official floating-base G1 27 body + 14 hand joints; XML inertias/colliders retained',
               'model_overrides': {'gravity': -9.81, 'floor_sliding_friction': 1., 'self_collision': False,
                                   'integrator': 'implicitfast', 'armature_and_torque_limits': 'actual Isaac runtime'},
               'limitations': ['Isaac USD and MuJoCo MJCF inertias/collision meshes not yet proven identical',
                               'explicit PD differs from Isaac implicit joint drives',
                               'virtual target event only; no physical punching bag or opponent',
                               'trial-level metric differs from Isaac 1000-env 500-frame autoreset evaluation'],
               'trials': records}
    suffix = '_smoke' if args.max_refs else ''
    (result_dir / f'{args.task}_zero_shot{suffix}.json').write_text(json.dumps(summary, indent=2))
    print('SIM2SIM_COMPLETE', args.task, summary['success_count'], '/', count, flush=True)
    renderer.close()


if __name__ == '__main__':
    main()
