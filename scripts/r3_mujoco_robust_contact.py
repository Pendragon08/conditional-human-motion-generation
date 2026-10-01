"""Preregistered frozen-policy stress and collidable-pad validation.

Reuses the verified original transfer adapter without changing that file or
the policies. Per-reference caches accelerate identical conditioning inputs
across scenarios; original reference values are never changed.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
from xml.sax.saxutils import quoteattr

import mujoco
import numpy as np
import torch

from r3_sim2sim_mujoco import ROOT, Transfer, VideoWriter, render_frame, sha

PROTOCOL = ROOT / 'configs/mujoco_robustness_contact_protocol.json'
OUT = ROOT / 'results/mujoco_robust_contact'
MEDIA = ROOT / 'media/mujoco_robust_contact'


class StressTransfer(Transfer):
    def __init__(self, task, device):
        super().__init__(task, device)
        old = self.model
        # The include and explicit mesh directory point to existing upstream
        # assets. No copies or alterations of its MJCF/meshes are needed.
        xml = f'''<mujoco model="frozen-policy-collidable-pad">
          <include file={quoteattr(str(self.scene_path))}/>
          <compiler meshdir={quoteattr(str(self.scene_path.parent / 'meshes'))}/>
          <worldbody><body name="r3_strike_pad" mocap="true" pos="10 0 1">
            <geom name="r3_strike_pad_geom" type="box" size="0.025 0.12 0.12"
              density="0" rgba="0.9 0.25 0.1 1" contype="4" conaffinity="1"
              friction="0.8 0.005 0.0001" solref="0.02 1" solimp="0.9 0.95 0.001"/>
          </body></worldbody></mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(xml)
        assert (self.model.nq, self.model.nv, self.model.nu) == (old.nq, old.nv, old.nu)
        for j in range(old.njnt):
            assert mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, j) == mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
        self.model.opt.timestep = old.opt.timestep
        self.model.opt.gravity[:] = old.opt.gravity
        self.model.opt.integrator = old.opt.integrator
        self.model.dof_armature[:] = old.dof_armature
        self.model.actuator_ctrllimited[:] = old.actuator_ctrllimited
        self.model.actuator_ctrlrange[:] = old.actuator_ctrlrange
        self.pad_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'r3_strike_pad')
        self.pad_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'r3_strike_pad_geom')
        self.pad_mocap = int(self.model.body_mocapid[self.pad_body])
        self.robot_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.endpoint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                         'right_ankle_roll_link' if task == 'Kick' else 'right_wrist_yaw_link')
        for g in range(self.model.ngeom):
            if g == self.pad_geom:
                continue
            if self.model.geom_bodyid[g] == 0:
                self.model.geom_contype[g], self.model.geom_conaffinity[g] = 2, 1
            elif self.model.geom_contype[g] or self.model.geom_conaffinity[g]:
                self.model.geom_contype[g], self.model.geom_conaffinity[g] = 1, 6
            self.model.geom_friction[g, 0] = 1.
        self.floor = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == 0]
        self.robot_bodies = [b for b in range(1, self.model.nbody) if b != self.pad_body]
        self.allowed = []
        for b in self.robot_bodies:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            if task == 'Kick' and name == 'right_ankle_roll_link':
                self.allowed.append(b)
            if task == 'Punch' and (name == 'right_wrist_yaw_link' or name.startswith('right_hand')):
                self.allowed.append(b)
        self.data = mujoco.MjData(self.model)
        self.base_constants = {k: getattr(self.model, k).copy() for k in
                               ['actuator_acc0', 'body_invweight0', 'dof_M0', 'dof_invweight0',
                                'body_subtreemass', 'light_poscom0']}
        self.base_friction = self.model.geom_friction.copy()
        self.base_mass = self.model.body_mass.copy()
        self.base_inertia = self.model.body_inertia.copy()
        self.base_limit = self.limit.copy()
        self.condition = {}
        self.normal = np.array([1., 0., 0.])
        self.event_time = 0.
        self.ref.state = lru_cache(maxsize=8192)(self.ref.state)
        self.provenance['stress_adapter_sha256'] = sha(__file__)
        self.provenance['protocol_sha256'] = sha(PROTOCOL)
        self.xml_sha256 = __import__('hashlib').sha256(xml.encode()).hexdigest()

    def configure(self, condition):
        self.condition = condition
        self.model.body_mass[:] = self.base_mass
        self.model.body_inertia[:] = self.base_inertia
        scale = condition.get('robot_mass_scale', 1.)
        self.model.body_mass[self.robot_bodies] *= scale
        self.model.body_inertia[self.robot_bodies] *= scale
        self.limit[:] = self.base_limit * condition.get('torque_limit_scale', 1.)
        self.model.actuator_ctrlrange[:, 0] = -self.limit
        self.model.actuator_ctrlrange[:, 1] = self.limit
        self.model.geom_friction[:] = self.base_friction
        if 'floor_friction' in condition:
            # MuJoCo combines equal-priority geom friction by max. Set both
            # members of floor/robot pairs, otherwise a 0.5 floor meets a 1.0
            # foot and yields 1.0 contact friction, not a low-friction test.
            ids = [g for g in range(self.model.ngeom) if g != self.pad_geom]
            self.model.geom_friction[ids, 0] = condition['floor_friction']
        self.model.geom_contype[self.pad_geom] = 4 if condition.get('pad') else 0
        self.model.geom_conaffinity[self.pad_geom] = 1 if condition.get('pad') else 0
        for key, value in self.base_constants.items():
            getattr(self.model, key)[:] = value
        if scale != 1. or condition.get('refresh_mass_constants'):
            mujoco.mj_setConst(self.model, self.data)

    def reset(self, i):
        super().reset(i)
        a = self.ref.a
        n = float(a['_motion_num_frames'][i])
        self.event_time = ((float(a['switch_idxs'][i])+1) / (n-1)) * float(a['_motion_lengths'][i])
        goal = self.ref.state(i, 0)['goal'].copy()
        vec = goal - self.data.qpos[:3]
        vec[2] = 0
        assert np.linalg.norm(vec) > 1e-5
        self.normal = vec / np.linalg.norm(vec)
        yaw = np.arctan2(self.normal[1], self.normal[0])
        shift = np.array(self.condition.get('pad_shift_world_m', [0, 0, 0]))
        self.pad_front = goal + shift
        self.data.mocap_pos[self.pad_mocap] = self.pad_front + self.normal*.025
        self.data.mocap_quat[self.pad_mocap] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        mujoco.mj_forward(self.model, self.data)
        self.push_impulse = np.zeros(3)
        self.contact_summary = {'any_contact': False, 'correct_effector_contact': False,
            'window_correct_contact': False, 'window_front_contact': False,
            'valid_strike_5n': False, 'valid_strike_10n': False, 'valid_strike_20n': False,
            'peak_any_normal_force_n': 0., 'peak_correct_normal_force_n': 0.,
            'peak_window_correct_force_n': 0., 'window_correct_impulse_ns': 0.,
            'correct_impulse_ns': 0., 'first_correct_contact_time_s': None,
            'max_pre_step_approach_velocity_m_s': 0., 'contact_substeps': 0}
        self.observed_floor_friction = set()
        initial = [float(self.data.contact[c].dist) for c in range(self.data.ncon)
                   if self.pad_geom in (self.data.contact[c].geom1, self.data.contact[c].geom2)]
        self.initial_penetration = min(initial, default=0.)

    def accumulate_contacts(self, pre_velocity):
        info = self.contact_summary
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            if self.pad_geom not in (contact.geom1, contact.geom2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, c, force)
            normal_force = max(0., float(force[0]))
            if normal_force <= 1e-6:
                continue
            other = contact.geom2 if contact.geom1 == self.pad_geom else contact.geom1
            body = int(self.model.geom_bodyid[other])
            info['any_contact'] = True
            info['peak_any_normal_force_n'] = max(info['peak_any_normal_force_n'], normal_force)
            if body not in self.allowed:
                continue
            info['correct_effector_contact'] = True
            info['contact_substeps'] += 1
            if info['first_correct_contact_time_s'] is None:
                info['first_correct_contact_time_s'] = float(self.data.time)
            info['peak_correct_normal_force_n'] = max(info['peak_correct_normal_force_n'], normal_force)
            info['correct_impulse_ns'] += normal_force*self.model.opt.timestep
            # contact.frame[0:3] points from geom1 to geom2. Orient from the
            # striker towards the pad before comparing to its front normal.
            normal = np.asarray(contact.frame[:3]) * (-1 if contact.geom1 == self.pad_geom else 1)
            front = float(np.dot(normal, self.normal)) >= .5
            window = abs(float(self.data.time)-self.event_time) <= .3
            if not window:
                continue
            info['window_correct_contact'] = True
            info['peak_window_correct_force_n'] = max(info['peak_window_correct_force_n'], normal_force)
            info['window_correct_impulse_ns'] += normal_force*self.model.opt.timestep
            approach = float(np.dot(pre_velocity, self.normal))
            info['max_pre_step_approach_velocity_m_s'] = max(info['max_pre_step_approach_velocity_m_s'], approach)
            if front:
                info['window_front_contact'] = True
                if approach >= .1:
                    for threshold in [5, 10, 20]:
                        if normal_force >= threshold:
                            info[f'valid_strike_{threshold}n'] = True

    def advance(self, action):
        self.targets[:] = self.hand_targets
        self.targets[self.body_actuators] = .5*action[:27] + self.default
        for _ in range(4):
            self.data.xfrc_applied[:] = 0.
            push_start = max(.2, self.event_time-.2)
            if push_start <= self.data.time < push_start+.2:
                force = np.array(self.condition.get('push_force_world_n', [0, 0, 0]), dtype=float)
                self.data.xfrc_applied[self.robot_body, :3] = force
                self.push_impulse += force*self.model.opt.timestep
            if self.condition.get('pad'):
                jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, jp, jr, self.point(), self.endpoint)
                velocity = jp @ self.data.qvel
            else:
                velocity = np.zeros(3)
            torque = self.kp*(self.targets-self.data.qpos[self.all_qadr])-self.kd*self.data.qvel[self.all_vadr]
            self.data.ctrl[:] = np.clip(torque, -self.limit, self.limit)
            mujoco.mj_step(self.model, self.data)
            for c in range(self.data.ncon):
                contact = self.data.contact[c]
                if contact.geom1 in self.floor or contact.geom2 in self.floor:
                    friction = float(contact.friction[0])
                    self.observed_floor_friction.add(round(friction, 8))
                    assert abs(friction-self.condition.get('floor_friction', 1.)) < 1e-8
            if self.condition.get('pad'):
                self.accumulate_contacts(velocity)
        self.last = action.copy()
        mujoco.mj_forward(self.model, self.data)


def detector_witness(t):
    t.configure({'id': 'engineered_detector_fixture', 'pad': True})
    t.reset(0)
    colliders = [g for g in range(t.model.ngeom) if t.model.geom_bodyid[g] in t.allowed
                 and (t.model.geom_contype[g] or t.model.geom_conaffinity[g])]
    assert colliders
    # A deliberately overlapping positive fixture checks the detector only.
    # It is never included among policy trials or called a policy success.
    g = colliders[0] if t.task == 'Kick' else next(g for g in colliders if t.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
    t.data.mocap_pos[t.pad_mocap] = t.data.geom_xpos[g].copy()
    t.data.mocap_quat[t.pad_mocap] = [1, 0, 0, 0]
    mujoco.mj_forward(t.model, t.data)
    t.accumulate_contacts(np.zeros(3))
    force = t.contact_summary['peak_correct_normal_force_n']
    positive = t.contact_summary['correct_effector_contact'] and force > 0
    t.model.geom_contype[t.pad_geom] = 0
    t.model.geom_conaffinity[t.pad_geom] = 0
    mujoco.mj_forward(t.model, t.data)
    negative = not any(t.pad_geom in (t.data.contact[c].geom1, t.data.contact[c].geom2) for c in range(t.data.ncon))
    assert positive and negative
    return {'engineered_detector_fixture_not_policy_trial': True, 'positive_contact_detected': positive,
            'positive_normal_force_n': force, 'collision_disabled_negative_passed': negative}


def no_collider_parity(t):
    baseline = Transfer(t.task, t.device)
    baseline.reset(0)
    t.configure({'id': 'nominal'})
    t.reset(0)
    worst_obs, worst_state, worst_action = 0., 0., 0.
    for k in range(30):
        ob1, ob2 = baseline.obs(0, k*.02), t.obs(0, k*.02)
        a1, a2 = baseline.action(ob1), t.action(ob2)
        worst_obs = max(worst_obs, float(np.max(np.abs(ob1-ob2))))
        worst_action = max(worst_action, float(np.max(np.abs(a1-a2))))
        baseline.advance(a1)
        t.advance(a2)
        worst_state = max(worst_state, float(np.max(np.abs(baseline.data.qpos-t.data.qpos))))
    assert worst_obs < 2e-4 and worst_action < 2e-4 and worst_state < 2e-4, (worst_obs, worst_action, worst_state)
    return {'control_steps': 30, 'max_obs_error': worst_obs, 'max_action_error': worst_action,
            'max_qpos_error': worst_state, 'passed': True}


def trial(t, i, condition, renderer, video):
    t.configure(condition)
    t.reset(i)
    valid_initialization = t.initial_penetration >= -.001
    if not valid_initialization:
        return {'reference_index': i, 'condition': condition['id'], 'initial_penetration_m': t.initial_penetration,
                'valid_initialization': False, 'failed_reason': 'initial_pad_overlap',
                'virtual_goal_reached': False, 'stable_goal_success': False, 'fallen': False,
                'survived_horizon': False, 'valid_strike_10n': False, 'stable_strike_success': False}
    writer = VideoWriter(video) if video else None
    success, fall, steps = False, False, 0
    goal_errors = []
    for k in range(500):
        action = t.action(t.obs(i, k*.02))
        assert np.isfinite(action).all()
        t.advance(action)
        now = (k+1)*.02
        state = t.ref.state(i, now)
        error = float(np.linalg.norm(t.point()-state['goal']))
        goal_errors.append(error)
        window = t.ref.state(i, now+.1)['closed'] and not t.ref.state(i, now-.1)['closed']
        if window and error < (.1 if t.task == 'Kick' else .05):
            success = True
        steps = k+1
        if writer and k % 2 == 0:
            writer.append_data(render_frame(renderer, t, state['goal']))
        if t.fallen():
            fall = True
            break
        assert np.isfinite(t.data.qpos).all()
    if writer:
        writer.close()
    survive = steps == 500 and not fall
    return {'reference_index': i, 'condition': condition['id'], 'valid_initialization': True,
            'initial_penetration_m': t.initial_penetration, 'executed_steps': steps,
            'survival_s': steps*.02, 'event_time_s': t.event_time, 'virtual_goal_reached': success,
            'stable_goal_success': success and survive, 'fallen': fall, 'survived_horizon': survive,
            'min_virtual_goal_error_m': min(goal_errors), 'push_impulse_world_ns': t.push_impulse.tolist(),
            'push_exposed': bool(np.linalg.norm(t.push_impulse) > 0),
            'observed_floor_contact_friction': sorted(t.observed_floor_friction),
            'stable_strike_success': t.contact_summary['valid_strike_10n'] and survive,
            'video': str(video.relative_to(ROOT)) if video else None, **t.contact_summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['Kick', 'Punch'], required=True)
    p.add_argument('--mode', choices=['witness', 'eval'], default='witness')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    t = StressTransfer(args.task, args.device)
    parity = t.verify()
    assert parity['passed']
    detector = detector_witness(t)
    physics_parity = no_collider_parity(t)
    if args.mode == 'witness':
        t.configure({'id': 'nominal'})
        t.reset(0)
        renderer = mujoco.Renderer(t.model, height=480, width=640)
        frame = render_frame(renderer, t, t.ref.state(0, 0)['goal'])
        from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER
        gpu = {'vendor': str(glGetString(GL_VENDOR)), 'renderer': str(glGetString(GL_RENDERER))}
        assert 'NVIDIA' in gpu['vendor'] and float(frame.std()) > 10
        renderer.close()
        witness = {'task': args.task, 'observation_parity': parity, 'detector': detector,
                   'collider_disabled_physics_parity': physics_parity, 'gpu_render': gpu,
                   'provenance': t.provenance, 'scene_wrapper_sha256': t.xml_sha256}
        (OUT/f'{args.task}_extension_witness.json').write_text(json.dumps(witness, indent=2))
        print('ROBUST_CONTACT_ENV_WITNESS', json.dumps(witness), flush=True)
        return
    count = protocol['references'][args.task]
    assert count == len(t.ref.a['_motion_lengths'])
    ids = list(range(args.shard, count, args.shards))
    renderer = mujoco.Renderer(t.model, height=480, width=640) if any(i in protocol['video_reference_ids'] for i in ids) else None
    result = {'task': args.task, 'shard': args.shard, 'shards': args.shards,
              'status': 'RUNNING', 'provenance': t.provenance, 'protocol': protocol, 'trials': []}
    path = OUT/f'{args.task}_shard_{args.shard}.json'
    start = time.monotonic()
    for i in ids:
        t.ref.state.cache_clear()
        for condition in protocol['conditions']:
            video = None
            if i in protocol['video_reference_ids'] and condition['id'] in protocol['video_conditions']:
                video = MEDIA/f"{args.task}_{condition['id']}_ref_{i:03d}.mp4"
            row = trial(t, i, condition, renderer, video)
            result['trials'].append(row)
            print('TRIAL', args.task, i, condition['id'], row['stable_goal_success'], row['stable_strike_success'], row['fallen'], flush=True)
        result['elapsed_wall_s'] = time.monotonic()-start
        path.write_text(json.dumps(result, indent=2))
    if renderer:
        renderer.close()
    result['status'] = 'COMPLETE'
    result['elapsed_wall_s'] = time.monotonic()-start
    path.write_text(json.dumps(result, indent=2))
    print('ROBUST_CONTACT_COMPLETE', args.task, args.shard, len(result['trials']), flush=True)


if __name__ == '__main__':
    main()
