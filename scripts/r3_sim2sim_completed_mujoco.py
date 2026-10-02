"""Frozen Jump/Sit/Pick/Bimanual-Pick transfer with real task scene collisions.

Uses the previously validated floating-base G1/PD adapter. No learning or IK.
Observation extensions execute the pinned official numerical function AST.
"""
import argparse
import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET
import numpy as np
import torch
import mujoco
import r3_sim2sim_mujoco as base

ROOT, DC, INTERFACE = base.ROOT, base.DC, base.INTERFACE
TASKS = {'Jump': '2026-10-01_14-01-43-Jump-a', 'Sit': '2026-10-01_16-54-27-Sit-a',
         'Pick': '2026-10-01_19-59-36-Pick-a', 'Bimanual-Pick': '2026-10-02_00-21-24-Bimanual-Pick-a'}
RESULT = ROOT/'results/sim2sim_completed_20261002'
MEDIA = ROOT/'media/sim2sim_completed_20261002'
base.policy_path = lambda task: DC/'Training/logs/rsl_rl/g1'/TASKS[task]/'exported/policy.pt'


def numerical(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for n in nodes:
        n.decorator_list = []
        n.returns = None
        for arg in n.args.args+n.args.kwonlyargs:
            arg.annotation = None
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


class CompletedTransfer(base.Transfer):
    def __init__(self, task):
        super().__init__(task, 'cuda:0')
        assert self.meta['root_reset_offset_z'] == .15
        self.term_sources = {}
        math_path = DC/'Training/source/isaaclab/isaaclab/utils/math.py'
        ns = numerical(math_path, ['quat_unique', 'quat_conjugate', 'quat_apply', 'quat_mul',
            'quat_from_matrix', '_sqrt_positive_part'], {'torch': torch})
        self.math = NS(**ns)
        self.functions = {}
        for term in self.meta['obs_terms']:
            if term['function'] in ['rel_pose_object', 'rel_pose_object_w_link', 'hand_pose',
                                    'target_orientation', 'hand_state_target', 'hand_state_target_1']:
                path, name = term['source'], term['function']
                ns = {'torch': torch, 'np': np, 'math_utils': self.math,
                      'SceneEntityCfg': lambda name, **kw: NS(name=name, **kw),
                      'ManagerBasedRLEnv': object, 'Articulation': object, 'RigidObject': object,
                      'slerp': self.ref.slerp}
                numerical(path, [name], ns)
                self.functions[term['name']] = ns[name]
                self.term_sources[path] = base.sha(path)
        self.term_sources[str(math_path)] = base.sha(math_path)
        self.provenance['completed_adapter_sha256'] = base.sha(__file__)
        self.provenance['observation_function_sources'] = self.term_sources
        old = self.model
        tree = ET.parse(self.scene_path)
        scenes = ROOT/'artifacts/sim2sim_mujoco/completed_scenes'
        scenes.mkdir(parents=True, exist_ok=True)
        for inc in tree.getroot().findall('include'):
            robot_xml = ET.parse(self.scene_path.parent/inc.get('file'))
            robot_xml.getroot().find('compiler').set('meshdir', str(self.scene_path.parent/'meshes'))
            robot_copy = scenes/'g1_training_robot.xml'
            robot_xml.write(robot_copy)
            inc.set('file', str(robot_copy))
        world = tree.getroot().find('worldbody')
        for s in self.meta['statics']:
            ET.SubElement(world, 'geom', name='task_'+s['name'], type='box',
                size=' '.join(str(x/2) for x in s['size']),
                pos=' '.join(map(str, s['position'])), quat=' '.join(map(str, s['quat'])),
                friction=f"{s['friction']} .005 .0001", rgba='0 .6 .2 1', contype='2', conaffinity='5')
        if 'object' in self.meta:
            o = self.meta['object']
            body = ET.SubElement(world, 'body', name='task_object', pos='0 0 1')
            ET.SubElement(body, 'freejoint', name='task_object_free')
            ET.SubElement(body, 'geom', name='task_object_geom', type='box',
                size=' '.join(str(x/2) for x in o['size']), mass=str(o['mass']),
                friction=f"{o['friction']} .005 .0001", rgba='0 .2 .8 1', contype='4', conaffinity='3')
        self.task_scene = scenes/f'{task}.xml'
        tree.write(self.task_scene)
        self.model = mujoco.MjModel.from_xml_path(str(self.task_scene))
        assert self.model.nu == old.nu
        self.model.opt.timestep = .005
        self.model.opt.gravity[:] = old.opt.gravity
        self.model.opt.integrator = old.opt.integrator
        for jid in self.ids:
            assert mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_JOINT, jid) == mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        self.model.dof_armature[:old.nv] = old.dof_armature
        self.model.actuator_ctrllimited[:] = old.actuator_ctrllimited
        self.model.actuator_ctrlrange[:] = old.actuator_ctrlrange
        for g in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            if name.startswith('task_'):
                continue
            if self.model.geom_bodyid[g] == 0:
                self.model.geom_contype[g], self.model.geom_conaffinity[g] = 2, 5
            elif self.model.geom_contype[g] or self.model.geom_conaffinity[g]:
                self.model.geom_contype[g], self.model.geom_conaffinity[g] = 1, 6
            self.model.geom_friction[g, 0] = 1.
        self.data = mujoco.MjData(self.model)
        self.body_ids = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in ['pelvis', 'torso_link', 'left_ankle_roll_link', 'right_ankle_roll_link',
                                   'left_wrist_yaw_link', 'right_wrist_yaw_link']}
        assert min(self.body_ids.values()) >= 0
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'task_object')
        if self.object_id >= 0:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'task_object_free')
            self.object_qadr = self.model.jnt_qposadr[j]
        self.provenance['task_scene_sha256'] = base.sha(self.task_scene)

    def extra_env(self, i, t, pos, quat, bodies):
        def tt(v): return torch.as_tensor(np.asarray(v), dtype=torch.float32).unsqueeze(0)
        data = NS(root_pos_w=tt(pos), root_quat_w=tt(quat),
            body_state_w=torch.stack([torch.cat([tt(bodies[n][0])[0], tt(bodies[n][1])[0]])
                for n in self.meta['body_names'] if n in bodies]).unsqueeze(0))
        # Only requested hand bodies are used by these official functions.
        names = [n for n in self.meta['body_names'] if n in bodies]
        class Scene(dict): pass
        scene = Scene(robot=NS(data=data)); scene.env_origins = torch.zeros(1, 3); scene.num_envs = 1
        class Lib:
            def get_motion_state(_self, ids, times):
                s = self.ref.state(int(ids[0]), float(times[0]))
                return {'grab_pos': tt(s['goal']), 'offsets': torch.zeros(1, 3),
                        'is_closed': torch.tensor([s['closed']])}
        env = NS(scene=scene, num_envs=1, device='cpu', motion_lib=Lib(), motion_ids=torch.tensor([i]),
                 episode_length_buf=torch.tensor([t/.02]), step_dt=.02, start_motion_times=torch.zeros(1))
        return env, names

    def build_obs(self, i, t, pos, quat, lin, ang, joints, velocities, last, bodies):
        values = {'base_lin_vel': lin, 'base_ang_vel': ang,
            'projected_gravity': base.rotation(quat).T@np.array([0., 0., -1.]),
            'joint_pos': joints-self.default, 'joint_vel': velocities, 'actions': last}
        env, names = self.extra_env(i, t, pos, quat, bodies)
        parts = []
        for term in self.meta['obs_terms']:
            name, function = term['name'], term['function']
            if name in values: v = values[name]
            elif function == 'target_ref': v = self.ref.target(i, t+term['time_offset'], pos, quat)
            elif function == 'rigid_body_mass': v = np.array([self.meta['object']['mass']])
            else:
                fn = self.functions[name]
                if function in ['hand_pose', 'rel_pose_object_w_link', 'target_orientation']:
                    side = 'left' if 'left' in name else 'right'
                    cfg = NS(name='robot', body_ids=[names.index(side+'_wrist_yaw_link')])
                    v = fn(env, asset_cfg=cfg)[0].numpy()
                else: v = fn(env)[0].numpy()
            v = np.asarray(v).ravel()
            if term['clip'] is not None: v = np.clip(v, *term['clip'])
            if term['scale'] is not None: v = v*term['scale']
            assert len(v) == term['dim'], (name, len(v), term['dim'])
            parts.append(v)
        return np.concatenate(parts).astype(np.float32)

    def bodies(self):
        return {n: (self.data.xpos[k].copy(), self.data.xquat[k].copy()) for n, k in self.body_ids.items()}

    def obs(self, i, t):
        q = self.data.qpos[3:7]
        return self.build_obs(i, t, self.data.qpos[:3], q, base.rotation(q).T@self.data.qvel[:3],
            self.data.qvel[3:6], self.data.qpos[self.qadr], self.data.qvel[self.vadr], self.last, self.bodies())

    def reset(self, i):
        super().reset(i)
        if self.object_id >= 0:
            o, s = self.meta['object'], self.ref.state(i, 0)
            pos = s['goal'].copy(); pos[2] = o['reset_height']; pos[:2] += o['reset_offset']
            self.data.qpos[self.object_qadr:self.object_qadr+7] = np.r_[pos, 1, 0, 0, 0]
            mujoco.mj_forward(self.model, self.data)

    def advance(self, action):
        for k, name in enumerate(['left_hand_action', 'right_hand_action']):
            h = self.meta['hand_actions'][name]
            values = h['close'] if action[27+k] < 0 else h['open']
            for joint, value in zip(h['joint_names'], values):
                self.targets[self.all_joint_names.index(joint)] = value
        self.targets[self.body_actuators] = .5*action[:27]+self.default
        excess = 0.
        for _ in range(4):
            tau = self.kp*(self.targets-self.data.qpos[self.all_qadr])-self.kd*self.data.qvel[self.all_vadr]
            excess = max(excess, float(np.maximum(abs(tau)-self.limit, 0).max()))
            self.data.ctrl[:] = np.clip(tau, -self.limit, self.limit)
            mujoco.mj_step(self.model, self.data)
        self.last = action.copy()
        mujoco.mj_forward(self.model, self.data)
        return excess

    def fallen(self):
        return bool(self.data.qpos[2] <= (.2 if self.task == 'Sit' else .3) or base.rotation(self.data.qpos[3:7])[2, 2] <= .5)

    def contacts(self):
        feet, hands, seat = set(), set(), False
        for c in self.data.contact[:self.data.ncon]:
            gs = [int(c.geom1), int(c.geom2)]
            names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or '' for g in gs]
            bn = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]) or '' for g in gs]
            if 'floor' in names:
                for n in bn:
                    if 'ankle_roll' in n: feet.add(n.split('_')[0])
            if 'task_object_geom' in names:
                for n in bn:
                    if 'hand' in n or 'wrist' in n: hands.add(n.split('_')[0])
            if 'task_seat' in names and any(n in ['pelvis', 'torso_link'] for n in bn): seat = True
        return feet, hands, seat

    def verify(self):
        rows = []
        lookup = {n: k for k, n in enumerate(self.meta['joint_names'])}
        jids = [lookup[n] for n in base.JOINTS]
        for n in range(30):
            s = dict(np.load(INTERFACE/f'{self.task}_snapshot_{n}.npz'))
            bodies = {b: (s['body_pos'][k], s['body_quat'][k]) for k,b in enumerate(self.meta['body_names'])}
            obs = self.build_obs(int(s['motion_id']), float(s['motion_time']), s['root_pos'], s['root_quat'],
                s['root_lin_vel_b'], s['root_ang_vel_b'], s['joint_pos'][jids], s['joint_vel'][jids], s['last_action'], bodies)
            action = self.action(obs)
            self.reset(int(s['motion_id']))
            self.data.qpos[:3], self.data.qpos[3:7] = s['root_pos'], s['root_quat']
            for k, name in enumerate(self.meta['joint_names']):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.data.qpos[self.model.jnt_qposadr[jid]] = s['joint_pos'][k]
                self.data.qvel[self.model.jnt_dofadr[jid]] = s['joint_vel'][k]
            self.data.qvel[:3] = base.rotation(s['root_quat'])@s['root_lin_vel_b']; self.data.qvel[3:6] = s['root_ang_vel_b']
            self.last = s['last_action'].copy(); mujoco.mj_forward(self.model, self.data)
            mapped = self.obs(int(s['motion_id']), float(s['motion_time']))
            rows.append({'snapshot': n, 'obs_error': float(abs(obs-s['obs']).max()),
                'action_error': float(abs(action-s['action']).max()), 'mapped_obs_error': float(abs(mapped-s['obs']).max())})
        return {'task': self.task, 'passed': all(max(x[k] for k in ['obs_error','action_error','mapped_obs_error']) < 2e-4 for x in rows),
                'obs_dim': len(obs), 'action_dim': len(action), 'snapshots': rows}


def main():
    p = argparse.ArgumentParser();p.add_argument('--task',choices=list(TASKS),required=True)
    p.add_argument('--mode',choices=['witness','eval'],default='witness');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    RESULT.mkdir(parents=True, exist_ok=True);MEDIA.mkdir(parents=True, exist_ok=True)
    t=CompletedTransfer(a.task);parity=t.verify()
    (RESULT/f'{a.task}_interface_parity.json').write_text(json.dumps(parity,indent=2))
    assert parity['passed'], parity
    print('COMPLETED_INTERFACE_PARITY_PASS',a.task,parity['obs_dim'],flush=True)
    renderer=mujoco.Renderer(t.model,height=480,width=640)
    from OpenGL.GL import glGetString,GL_VENDOR,GL_RENDERER
    gl={'vendor':str(glGetString(GL_VENDOR)),'renderer':str(glGetString(GL_RENDERER))};assert 'NVIDIA' in gl['vendor']
    t.reset(0);before=t.data.qpos.copy()
    for k in range(10):t.advance(t.action(t.obs(0,k*.02)))
    frame=base.render_frame(renderer,t,t.ref.state(0,0)['goal'])
    witness={'task':a.task,'gl':gl,'physics_time':t.data.time,'state_changed':bool(abs(t.data.qpos-before).max()>1e-6),
        'finite':bool(np.isfinite(t.data.qpos).all()),'render_std':float(frame.std()),'cuda_device':str(next(t.policy.parameters()).device),
        'provenance':t.provenance,'environment':'warm reuse hash91455383; MuJoCo CPU dynamics/CUDA policy/NVIDIA EGL'}
    assert witness['state_changed'] and witness['finite'] and witness['render_std']>10
    (RESULT/f'{a.task}_environment_witness.json').write_text(json.dumps(witness,indent=2))
    print('COMPLETED_SIM2SIM_WITNESS',a.task,flush=True)
    if a.mode=='witness':renderer.close();return
    start=time.monotonic();records=[]
    for i in range(t.meta['reference_count']):
        t.reset(i);video=MEDIA/f'{a.task}_reference_{i:03d}.mp4' if i<2 else None
        writer=base.VideoWriter(video) if video else None
        initial_obj=float(t.data.xpos[t.object_id,2]) if t.object_id>=0 else None
        success=False;fall=False;air=0;land=0;air_seen=False;ground_seen=False;takeoff_frames=0;seat_hold=0;lift_hold=0
        best_seat=0;best_lift=0;hand_frames={'left':0,'right':0,'both':0};jerrors=[];goalerrors=[];maxlift=0
        for k in range(500):
            action=t.action(t.obs(i,k*.02));assert np.isfinite(action).all()
            t.advance(action);now=(k+1)*.02;s=t.ref.state(i,now);feet,hands,seat=t.contacts()
            fall=t.fallen();jerrors.append(float(np.mean((t.data.qpos[t.qadr]-s['joints'])**2)))
            if a.task in ['Jump','Sit']:
                endpoint=t.body_ids['torso_link' if a.task=='Jump' else 'pelvis']
                err=float(np.linalg.norm(t.data.xpos[endpoint]-s['goal']));goalerrors.append(err)
                window=t.ref.state(i,now+.1)['closed'] and not t.ref.state(i,now-.1)['closed']
                success |= bool(err<.05 and window)
            if not feet:
                air+=1;land=0
                if ground_seen:air_seen=True;takeoff_frames+=1
            else:
                ground_seen=True
                land=land+1 if air_seen and not fall else 0
            seat_hold=seat_hold+1 if seat and not fall else 0;best_seat=max(best_seat,seat_hold)
            if t.object_id>=0:
                z=float(t.data.xpos[t.object_id,2]);lift=z-initial_obj;maxlift=max(maxlift,lift)
                success |= bool(z>t.meta['object']['success_height'] and s['closed'])
                for side in ['left','right']:hand_frames[side]+=side in hands
                hand_frames['both']+=len(hands)==2
                contact=('right' in hands) if a.task=='Pick' else len(hands)==2
                lift_hold=lift_hold+1 if lift>=.05 and contact and not fall else 0;best_lift=max(best_lift,lift_hold)
            if writer and k%2==0:writer.append_data(base.render_frame(renderer,t,s['goal']))
            assert np.isfinite(t.data.qpos).all()
            if fall:break
        if writer:writer.close()
        row={'reference_index':i,'official_criterion_proxy_success':success,'fallen':fall,'steps':k+1,
            'survival_s':(k+1)*.02,'full_horizon_survival':k==499 and not fall,
            'joint_tracking_rmse_rad':float(np.sqrt(np.mean(jerrors))),
            'min_goal_distance_m':min(goalerrors) if goalerrors else None,
            'airborne_control_frames':air,'post_initial_ground_contact_airborne_frames':takeoff_frames,
            'landing_stability_s':land*.02,
            'seat_contact_max_hold_s':best_seat*.02,'initial_object_height_m':initial_obj,
            'max_object_lift_from_reset_m':maxlift,'hand_object_contact_frames':hand_frames,
            'contact_lift_max_hold_s':best_lift*.02,'video':str(video) if video else None}
        row['additional_task_check_success']=(takeoff_frames>=2 and land>=50 and not fall) if a.task=='Jump' else (best_seat>=50 and not fall) if a.task=='Sit' else best_lift>=50 and not fall
        records.append(row);print('TRIAL',a.task,i,success,fall,row['additional_task_check_success'],flush=True)
    n=len(records);summary={'task':a.task,'status':'COMPLETE','policy_frozen':True,'learning':False,'extra_ik':False,
        'num_references':n,'protocol':'one deterministic start-at-zero episode per original reference; 500 control steps; no autoreset; failures remain denominator',
        'official_proxy_success_count':sum(x['official_criterion_proxy_success'] for x in records),
        'additional_task_check_success_count':sum(x['additional_task_check_success'] for x in records),
        'fall_count':sum(x['fallen'] for x in records),'full_horizon_survival_count':sum(x['full_horizon_survival'] for x in records),
        'elapsed_wall_s':time.monotonic()-start,'physics_dt_s':.005,'policy_dt_s':.02,
        'reference_npz_sha256':base.sha(INTERFACE/f'{a.task}_references.npz'),'provenance':t.provenance,
        'task_scene':{'statics':t.meta['statics'],'object':t.meta.get('object')},
        'limitations':['MJCF and USD robot inertias/collision shapes not proven identical; explicit PD differs from Isaac implicit drives',
            'trial protocol differs from Isaac 1000-env autoreset evaluation; not directly comparable success percentages',
            'primitive training task colliders used, not photorealistic play assets',
            'one seed0 sampled object mass held fixed; no domain-randomization distribution evaluation',
            'official pickup absolute-height criterion may already hold at reset; separate contact-plus-relative-lift test reported',
            'single-reference tasks have n=1, not a population success estimate'], 'trials':records}
    (RESULT/f'{a.task}_zero_shot.json').write_text(json.dumps(summary,indent=2));renderer.close()
    print('COMPLETED_SIM2SIM_EVAL',a.task,n,flush=True)


if __name__=='__main__':main()
