"""Frozen SuperMimic policy in the four dynamic-object PhysX tasks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PICK = ROOT / 'stage3_pick29_20261103'
SUP = ROOT / 'stage3_supported_references_20261104'
NATIVE = ROOT / 'stage3_supermimic29_20261103/native_universal_20261104'
SOURCE = NATIVE / 'source/supermimic'
MODEL = NATIVE / 'formal2000_sequence_v1/main_resume_v1/logs/last.pt'
LIBRARY = SUP / 'mixed2000_final_v1/native_motion_manifest.json'
MODEL_SHA = '571b207875610946f1065bf25bdf1dac78cb8c37f1691fc001324b1da64776ee'
LIBRARY_SHA = '6d35b617780968b45ef9d05c2014df17708d6b45bb7158438a13ec6c49df36cf'
TASKS = ('pick', 'pour', 'bimanual_box_lift', 'box_push_pull')
SEED = 20261105
FLOOR = 8 * 1024**3 + 456 * 1024**2


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    with Path(path).open('x') as file:
        json.dump(data, file, indent=2, allow_nan=False)


def sha(path):
    with Path(path).open('rb') as file:
        h = hashlib.sha256()
        for block in iter(lambda: file.read(4 * 1024**2), b''):
            h.update(block)
        return h.hexdigest()


def policy_process(selection):
    """Run the original policy, MDP and MotionLib in their original Python environment."""
    import numpy as np
    import torch
    import supermimic.networks as networks
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, OBS_DIM
    from supermimic.motion import MotionLib
    from supermimic.robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list, _actuators
    from supermimic.rotations import mat_to_quat, quat_to_mat

    device = 'cuda:0'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(2 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    rows = read(selection)
    kin = G1Kinematics(device)
    motion = MotionLib([row['csv'] for row in rows], kin, device)
    assert motion.names == [Path(row['csv']).stem for row in rows]
    assert sha(MODEL) == MODEL_SHA
    saved = torch.load(MODEL, map_location='cpu', weights_only=True)
    args = saved['args']
    future, history = args['goal_frames'], args['hist']
    assert (future, history) == (10, 10)
    networks.AMP_DTYPE = torch.bfloat16 if args.get('amp') == 'bf16' else None
    policy = networks.Policy(OBS_DIM, 73 * future, hist=history).to(device)
    policy.load_state_dict(saved['pi'], strict=True)
    del saved
    policy.eval().requires_grad_(False)
    initial_hash = policy_hash(policy)
    default = default_joint_pos_list()
    mdp = MDP(kin, motion, torch.tensor(default, device=device))
    f = motion.clip_start
    p, rotation, v, w, q, qd = motion.ref_state(f)
    q = q.clamp(kin.q_lo, kin.q_hi)
    initial = torch.cat((p, mat_to_quat(rotation), v, w, q, qd), dim=1)
    motors = {}
    for cfg in _actuators().values():
        motors[cfg.joint_names_expr[0]] = {
            'effort_limit_sim': cfg.effort_limit_sim, 'velocity_limit_sim': cfg.velocity_limit_sim,
            'stiffness': cfg.stiffness, 'damping': cfg.damping, 'armature': cfg.armature,
        }
    print(json.dumps({
        'names': motion.names, 'joints': G1_JOINT_NAMES, 'clip_lengths': motion.clip_len.tolist(),
        'initial': initial.tolist(), 'default_q': default, 'motors': motors,
        'scale': [action_scale(name) for name in G1_JOINT_NAMES],
        'native_policy_state_sha256': initial_hash,
    }), flush=True)
    count = len(rows)
    xi = torch.zeros(count, 33, device=device)
    xi[:, 0] = 1
    prev = torch.zeros(count, 29, device=device)
    hist = None
    with torch.inference_mode():
        while data := sys.stdin.buffer.read(count * 71 * 4):
            assert len(data) == count * 71 * 4
            x = torch.from_numpy(np.frombuffer(data, dtype='<f4').copy().reshape(count, 71)).to(device)
            assert torch.isfinite(x).all()
            state = (x[:, :3], quat_to_mat(x[:, 3:7]), x[:, 7:10], x[:, 10:13], x[:, 13:42], x[:, 42:])
            obs = mdp.obs(state, prev, xi, f)
            hist = obs[:, None].repeat(1, history, 1) if hist is None else torch.cat((hist[:, 1:], obs[:, None]), 1)
            prev = policy(hist, motion.goal(f, future))
            assert prev.shape == (count, 29) and torch.isfinite(prev).all()
            sys.stdout.buffer.write(prev.cpu().numpy().astype('<f4', copy=False).tobytes())
            sys.stdout.buffer.flush()
            f = torch.minimum(f + 1, motion.clip_end)
    assert policy_hash(policy) == initial_hash and sha(MODEL) == MODEL_SHA


def policy_hash(policy):
    h = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def evaluation(args):
    import numpy as np
    import torch
    import isaaclab.envs.mdp as isaac_mdp
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, RewardTermCfg
    from isaaclab.utils import configclass
    import stage3_pick29_env as base
    from stage3_supported_ppo_core_v1 import empty_milestones
    from stage3_supported_ppo_env_v1 import SupportedSceneCfg, contact_pairs, make_cfg, measured

    assert shutil.disk_usage('/data').free >= FLOOR
    assert sha(LIBRARY) == LIBRARY_SHA and sha(MODEL) == MODEL_SHA
    assert args.device == 'cuda:0' and args.num_envs in (4, 100)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    device, task, count = args.device, args.task, args.num_envs
    library = read(LIBRARY)
    rows = [row for row in library['records'] if row['task'] == task]
    assert len(rows) == 500
    rng = np.random.default_rng(SEED)
    rows = [rows[i] for i in rng.permutation(len(rows))[:count]]
    selected = args.output / 'selection.json'
    write(selected, [{key: row[key] for key in ('candidate_id', 'csv', 'csv_sha256', 'source', 'source_sha256')} for row in rows])
    hands, positions, quats, grasp_times = [], [], [], []
    for row in rows:
        assert sha(row['csv']) == row['csv_sha256'] and sha(row['source']) == row['source_sha256']
        hands.append(np.load(row['source'], allow_pickle=False)['hand_closed'].copy())
        positions.append(row['object_metadata']['initial_position'])
        quats.append(row['object_metadata']['initial_quaternion_wxyz'])
        grasp_times.append(float(row['metadata']['grasp_time_s']) if task == 'pick' else 2.)
    hands = torch.tensor(np.stack(hands), device=device)
    assert hands.shape == (count, rows[0]['source_frames'], 2)
    object_positions = torch.tensor(positions, device=device)
    object_quats = torch.tensor(quats, device=device)
    target_step = torch.tensor(np.ceil(np.array(grasp_times) * 50 - 1e-6), device=device, dtype=torch.long)
    sides = [1] if task in ('pick', 'pour') else [0, 1]
    if task == 'pick':
        cfg = base.Pick29EnvCfg()
        cfg.scene = SupportedSceneCfg(num_envs=count, env_spacing=3.)
        cfg.sim.device = device
        scene = {'object_size': base.SCENE['object_size']}
    else:
        kin_report = read(rows[0]['kinematic_report'])
        scene = read(Path(kin_report['sample']).parent / 'report.json')['scene']
        scene_digest = hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert all(row['metadata']['scene_sha256'] == scene_digest for row in rows)
        scene_input = args.output / 'scene_input.json'
        write(scene_input, {'task': task, 'scene': scene})
        cfg = make_cfg(task, scene_input, device, count)
    cfg.seed = SEED
    assert cfg.decimation * cfg.sim.dt == .02
    assert not cfg.scene.object.spawn.rigid_props.kinematic_enabled

    # The target is specified before each rollout. Push/pull reports both engagements.
    offsets = [[0., 0., .04]]
    if task == 'bimanual_box_lift':
        width = scene['object_size'][1] / 2 + .025
        offsets = [[0., width, 0.], [0., -width, 0.]]
    elif task == 'box_push_pull':
        front = -scene['object_size'][0] / 2 - .025
        offsets = [[front, .105, 0.], [front, -.105, 0.]]
    targets = [object_positions[:, None] + torch.tensor(offsets, device=device)]
    schedule = [target_step]
    names = ['initial_interaction']
    if task == 'box_push_pull':
        targets.append(targets[0] + torch.tensor([.10, 0., 0.], device=device))
        schedule.append(torch.full_like(target_step, 260))
        names.append('pull_recontact')
    errors = torch.full((count, len(names)), torch.nan, device=device)
    contacts_at_event = torch.zeros_like(errors, dtype=torch.bool)
    first = torch.ones(count, dtype=torch.bool, device=device)
    completed_steps = torch.zeros_like(target_step)
    success = torch.zeros_like(first)
    seen_contact = torch.zeros_like(first)

    spec = read(NATIVE / 'environment_spec.json')['spec']
    # The native runtime uses a separate Python ABI. Exchange only physical state and policy actions.
    env = {key: value for key, value in os.environ.items()
           if key not in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'ISAAC_PATH', 'EXP_PATH')}
    env.update(PYTHONPATH=':'.join(spec['module_paths']), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    process = None
    sim_env = None
    with (args.output / 'policy_stderr.log').open('x') as log:
        try:
            process = subprocess.Popen(
                [spec['python'], '-u', str(Path(__file__).resolve()), '--native-policy', str(selected)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env,
            )
            line = process.stdout.readline()
            if not line:
                raise RuntimeError('Native policy startup failed. See policy_stderr.log.')
            native = json.loads(line)
            assert native['joints'] == base.BODY_JOINTS
            initial = torch.tensor(native['initial'], device=device)
            clip_lengths = torch.tensor(native['clip_lengths'], device=device)
            assert initial.shape == (count, 71)
            assert torch.all(clip_lengths > torch.stack(schedule).amax(0))
            joints = native['joints']
            hand_actuator = cfg.scene.robot.actuators['hands']
            cfg.scene.robot.actuators = {'hands': hand_actuator, **{
                name: ImplicitActuatorCfg(joint_names_expr=[name], **native['motors'][name]) for name in joints
            }}
            cfg.actions.body.use_default_offset = False
            cfg.actions.body.offset = dict(zip(joints, native['default_q']))
            cfg.actions.body.scale = dict(zip(joints, native['scale']))

            def setup(env, env_ids=None):
                robot = env.scene['robot']
                env._body29_ids, joint_names = robot.find_joints(joints, preserve_order=True)
                assert joint_names == joints and len(robot.joint_names) == 43
                env._wrists = robot.find_bodies(['left_wrist_yaw_link', 'right_wrist_yaw_link'], preserve_order=True)[0]
                env._reference_bank = argparse.Namespace(scene=scene)
                env._milestones = empty_milestones(count, device)
                env._object_initial = object_positions
                env._task_cache_step = -1

            def reset(env, env_ids):
                if env_ids is None:
                    env_ids = torch.arange(count, device=device)
                isaac_mdp.reset_scene_to_default(env, env_ids)
                robot = env.scene['robot']
                x = initial[env_ids]
                pose = torch.cat((x[:, :3] + env.scene.env_origins[env_ids], x[:, 3:7]), dim=1)
                robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
                robot.write_root_link_velocity_to_sim(x[:, 7:13], env_ids=env_ids)
                robot.write_joint_state_to_sim(x[:, 13:42], x[:, 42:], joint_ids=env._body29_ids, env_ids=env_ids)
                obj = env.scene['object']
                state = obj.data.default_root_state[env_ids].clone()
                state[:, :3] = object_positions[env_ids] + env.scene.env_origins[env_ids]
                state[:, 3:7] = object_quats[env_ids]
                state[:, 7:] = 0
                obj.write_root_state_to_sim(state, env_ids=env_ids)
                for value in env._milestones.values():
                    value[env_ids] = False
                env._task_cache_step = -1

            def capture(env):
                # Reward callbacks read the post-step simulation state before auto-reset.
                steps = env.episode_length_buf
                wrist = env.scene['robot'].data.body_pos_w[:, env._wrists] - env.scene.env_origins[:, None]
                if task == 'pick':
                    contacts = contact_pairs(env)
                    z = env.scene['object'].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
                    passed = (z > base.SCENE['success_height']) & (steps >= target_step)
                else:
                    contacts = measured(env)['contacts']
                    passed = env._milestones['success']
                success.logical_or_(first & passed)
                seen_contact.logical_or_(first & contacts[:, sides].all(-1))
                for i, (when, target) in enumerate(zip(schedule, targets)):
                    at_event = first & (steps == when)
                    distance = (wrist[:, sides] - target).norm(dim=-1).mean(-1)
                    errors[:, i] = torch.where(at_event, distance, errors[:, i])
                    contacts_at_event[:, i] |= at_event & contacts[:, sides].all(-1)
                return torch.zeros(count, device=device)

            @configclass
            class CaptureCfg:
                capture_state = RewardTermCfg(func=capture, weight=1.)

            cfg.events.initialize = EventTermCfg(func=setup, mode='startup')
            cfg.events.reset = EventTermCfg(func=reset, mode='reset')
            cfg.rewards = CaptureCfg()
            cfg.terminations.reference_end = None
            sim_env = ManagerBasedRLEnv(cfg=cfg)
            sim_env.reset()
            robot = sim_env.scene['robot']
            pos = sim_env.scene['object'].data.root_pos_w - sim_env.scene.env_origins
            assert torch.allclose(pos, object_positions, atol=2e-5, rtol=0.)
            assert sim_env.action_manager.total_action_dim == 31
            print(f'{task}: dynamic PhysX initialized, {count} first episodes', flush=True)
            with torch.inference_mode():
                for step in range(int(clip_lengths.max()) - 1):
                    if step % 50 == 0:
                        assert shutil.disk_usage('/data').free >= FLOOR
                    data = robot.data
                    pose = data.root_link_pose_w
                    x = torch.cat((pose[:, :3] - sim_env.scene.env_origins, pose[:, 3:],
                                   data.root_link_vel_w, data.joint_pos[:, sim_env._body29_ids],
                                   data.joint_vel[:, sim_env._body29_ids]), dim=1)
                    assert x.shape == (count, 71)
                    # A fallen episode is masked out of all later metrics.
                    x = torch.where(first[:, None], x, initial)
                    process.stdin.write(x.cpu().numpy().astype('<f4', copy=False).tobytes())
                    process.stdin.flush()
                    answer = process.stdout.read(count * 29 * 4)
                    if len(answer) != count * 29 * 4:
                        raise RuntimeError('Native policy stopped during rollout. See policy_stderr.log.')
                    body = torch.tensor(np.frombuffer(answer, dtype='<f4').copy().reshape(count, 29), device=device)
                    hand = 1. - 2. * hands[:, min(step * 2 // 5, hands.shape[1] - 1)].float()
                    _, _, terminated, truncated, _ = sim_env.step(torch.cat((body, hand), dim=1))
                    completed_steps += first.long()
                    first &= ~(terminated | truncated | (completed_steps >= clip_lengths - 1))
                    if not bool(first.any()):
                        break
            process.stdin.close()
            assert process.wait(timeout=60) == 0
            assert sha(MODEL) == MODEL_SHA
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            if sim_env is not None:
                sim_env.close()

    def mean(x):
        return float(x.mean()) if x.numel() else None

    observed = errors.isfinite()
    complete = observed.all(-1)
    successes = int(success.sum())
    event_summary = {
        name: {'mean_error_m': mean(errors[:, i][observed[:, i]]), 'measured_episodes': int(observed[:, i].sum()),
               'actual_contact_at_measurement': int(contacts_at_event[:, i].sum())}
        for i, name in enumerate(names)
    }
    report = {
        'status': 'frozen_policy_task_evaluation_completed', 'task': task, 'episode_count': count,
        'integration_only': count < 100, 'success_count': successes, 'success_ratio': successes / count,
        'control_error_m': mean(errors[complete].mean(-1)), 'control_error_episode_count': int(complete.sum()),
        'interaction_events': event_summary, 'episodes_with_required_hand_contact': int(seen_contact.sum()),
        'interaction_error_definition': 'L2 from simulated wrist(s) to specified world target(s) at scheduled interaction step; average hands, then events, then episodes with all events observed',
        'missing_error_definition': 'Episodes terminating before a required measurement stay in success denominator; missing errors are null.',
        'success_definition': {
            'pick': 'actual object center above 0.95 m at or after the grasp step (height criterion)',
            'pour': 'actual right hand contact, lift, tilt > 60 degrees, upright restore and place/release; bottle proxy without fluid',
            'bimanual_box_lift': 'actual bimanual contact, lift > 0.10 m, carry > 0.08 m, place and release',
            'box_push_pull': 'actual push > 0.08 m, release, recontact and return within 0.025 m',
        }[task],
        'sampling': 'seeded sample without replacement from 500 trained-reference scenes with paired randomized object XY',
        'seed': SEED, 'physics': 'Isaac Sim 4.5 PhysX, dynamic objects, no object state writes after first-episode reset',
        'body_controller': 'frozen reference-conditioned SuperMimic, native observation/history/goal, native joint target scale and motor parameters',
        'hands': 'existing binary hand actuators execute reference open/close schedule',
        'optimizer_updates': 0, 'checkpoint_sha256': MODEL_SHA, 'library_sha256': LIBRARY_SHA,
        'policy_state_sha256': native['native_policy_state_sha256'],
        'asset_sha256': sha(base.ASSET),
        'episodes': [
            {'candidate_id': row['candidate_id'], 'object_initial_xyz_m': positions[i],
             'success': bool(success[i]), 'steps': int(completed_steps[i]),
             'control_error_m': {name: float(errors[i, j]) if bool(observed[i, j]) else None for j, name in enumerate(names)}}
            for i, row in enumerate(rows)
        ],
    }
    write(args.output / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('task', 'success_ratio', 'control_error_m', 'control_error_episode_count')}), flush=True)


def main():
    if sys.argv[1:2] == ['--native-policy']:
        policy_process(Path(sys.argv[2]))
        return
    sys.path[:0] = [str(PICK / 'src'), str(SUP / 'src')]
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--num-envs', type=int, choices=(4, 100), required=True)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    app = None
    try:
        app = AppLauncher(args).app
        evaluation(args)
    except BaseException:
        write(args.output / 'failure.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        if app is not None:
            app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()