"""Actual full-clip physical screening with the pinned external ScaleBFM tracker.

This is an auxiliary data-quality control, NOT the paper's trained PhyGile GMT
filter. No source motion is removed and no author curriculum labels are created.
Restricted references/executions stay private; public summaries contain hashes.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'STARTING', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'filter_role': 'external_pretrained_tracker_auxiliary_screen',
        'paper_phygile_gmt_filter_reproduced': False, 'generator_training_allowed_by_this_receipt': False,
        'new_optimizer_updates': 0, 'source_data_deleted': False, 'gpu_used': False, 'records': [],
        'thresholds': {'pelvis_z_error_m': .3, 'torso_projected_gravity_l2': .8,
            'end_effector_z_error_m': .3, 'full_clip_mean_mpjpe_m': .1},
        'threshold_provenance': 'First three from paper tracking evaluation; full-clip MPJPE0.1m is '
            'an explicit reconstruction choice, not a published dataset-filter threshold.'}
    save(a.output/'summary.json', report)
    bound = {}

    def bind(path):
        path = Path(path).resolve(strict=True)
        bound[str(path)] = sha(path)
        return path

    try:
        torch.set_num_threads(4)
        import mujoco
        upstream = a.code_root/'upstream'
        sys.path[:0] = [str(a.code_root/'scripts'), str(upstream), str(upstream/'deploy'),
                        str(a.phygile_code/'reimplementation')]
        from codec262 import G1Codec, quat_matrix
        from motion_clock import resample_native
        from scalebridge.utils.config import load_yaml, deep_merge, to_config
        from scalebridge.env.motion_tracking import MotionTrackingEnv
        from torch_bfm_agent import TorchBFMAgent
        checkpoint = bind(a.runtime_root/'weights/ScaleBFM_m_model_22200.pt')
        if sha(checkpoint) != '88d5a79946c03ed25503f48b2af71d16290844ef066ca9b6c8fa8dc3837422e3':
            raise ValueError('Unexpected external tracking checkpoint')
        bind(__file__)
        bind(a.code_root/'scripts/torch_bfm_agent.py')
        for f in sorted(upstream.rglob('*.py')):
            bind(f)
        for name in ['codec262.py', 'motion_clock.py']:
            bind(a.phygile_code/'reimplementation'/name)
        manifest_path = bind(a.manifest)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('sample_clock_gate', {}).get('status') != 'NUMERIC_CLOCK_CONSISTENCY_ONLY':
            raise ValueError('Audited clock-gated pairs required')
        entries = [(s, e) for s, rows in manifest['splits'].items() for e in rows]
        if a.limit:
            entries = entries[a.start:a.start+a.limit]
        else:
            entries = entries[a.start:]
        if not entries:
            raise ValueError('No selected pairs')
        codec = G1Codec(bind(a.phygile_code/'configs/g1_codec_skeleton.json'))
        config_dir = upstream/'deploy/scalebridge/config'
        for f in config_dir.rglob('*.yaml'):
            bind(f)
        base = load_yaml(config_dir/'agent/base_agent.yaml')
        specific = load_yaml(config_dir/'agent/bfm_agent.yaml')
        agent_cfg = deep_merge(base.config, specific.config)
        agent_cfg.checkpoint = str(checkpoint)
        agent_cfg.control_mode = 7
        agent = TorchBFMAgent(agent_cfg, 'cpu')
        metadata = agent.get_meta_data()
        policy_names = tuple(metadata['joint_names'])
        if set(policy_names) != set(codec.joint_names):
            raise ValueError('Joint-name sets differ; refusing implicit actuator ordering')
        order = [codec.joint_names.index(n) for n in policy_names]
        report.update(checkpoint_sha256=sha(checkpoint), control_mode=7, selected_sequences=len(entries),
                      seeds=a.seeds, source_manifest_sha256=sha(manifest_path), device='cpu',
                      torch=torch.__version__, mujoco=mujoco.__version__,
                      reference_forcing=False, initialization='exact first reference pose and velocities',
                      timing='post-step/pre-reset state against current commanded reference frame; no lag fitting',
                      measurement='separate MjData kinematics from the current integrated qpos; live state unmodified')
        policy_hash = hashlib.sha256()
        for name, tensor in agent.policy.state_dict().items():
            policy_hash.update(name.encode());policy_hash.update(tensor.cpu().contiguous().numpy().tobytes())
        report['policy_parameter_digest_before'] = policy_hash.hexdigest()
        prepared = a.output/'private_references'
        prepared.mkdir()
        for index, (split, e) in enumerate(entries, a.start):
            feature = bind(manifest_path.parent/e['features'])
            sidecar = bind(manifest_path.parent/e['codec_metadata'])
            if sha(feature) != e['sha256'] or not e['sample_clock_gate']['eligible']:
                raise ValueError('Pair identity/clock gate changed')
            x = np.load(feature, allow_pickle=False)
            side = json.loads(sidecar.read_text(encoding='utf-8'))
            native, _, _ = codec.decode(x, side, project=False, restore_world=True)
            times = np.arange(len(native))/float(side['fps'])
            query = np.arange(int(np.floor(times[-1]*50+1e-9))+1)/50
            native = resample_native(native, times, query)
            ref_file = prepared/f'pair_{index:03d}.npz'
            np.savez_compressed(ref_file, fps=np.array(50.), root_pos=native[:, :3],
                root_rot=native[:, 3:7], dof_pos=native[:, 13:42][:, order],
                dof_vel=native[:, 42:71][:, order])
            for seed in a.seeds:
                torch.manual_seed(seed);np.random.seed(seed)
                episode = a.output/f'pair_{index:03d}_seed_{seed}'
                episode.mkdir()
                row = {'index': index, 'split': split, 'features_sha256': sha(feature),
                    'source_group_sha256': hashlib.sha256(e['source_group'].encode()).hexdigest(),
                    'mirrored': e['mirrored'], 'seed': seed, 'reference_frames': len(native),
                    'reference_sha256': sha(ref_file), 'status': 'STARTING',
                    'feature_fps': float(side['fps']), 'control_fps': 50.,
                    'omitted_tail_s': float(times[-1]-query[-1])}
                save(episode/'receipt.json', row)
                asset = load_yaml(config_dir/'asset/g1_29dof.yaml')
                asset.xml_path = str((upstream/'deploy'/asset.xml_path).resolve())
                xml = bind(asset.xml_path)
                sb = load_yaml(config_dir/'simulator/base_simulator.yaml')
                ss = load_yaml(config_dir/'simulator/mujoco_simulator.yaml')
                scfg = deep_merge(sb.config, ss.config)
                scfg.asset = asset;scfg.headless = True;scfg.record_video = False
                scfg.marker = False;scfg.joystick = False
                ecfg = load_yaml(config_dir/'env/base_env.yaml').config
                ecfg.simulator = to_config({'_target_': ss['_target_'], 'config': scfg})
                ecfg.motion_path = str(ref_file)
                ecfg.observation = load_yaml(config_dir/'observation/bfm.yaml')
                ecfg.future_idx = metadata['future_idx']
                ecfg.reference_forcing = False
                env = MotionTrackingEnv(ecfg, copy.deepcopy(metadata), 'cpu')
                sim = env.simulator
                try:
                    model, data = sim.mujoco_model, sim.mujoco_data
                    qpos_ids = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in codec.joint_names]
                    if len(set(qpos_ids)) != 29 or min(qpos_ids) < 7:
                        raise ValueError('Native joint-to-XML mapping invalid')
                    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in codec.body_names]
                    if min(body_ids) < 0:
                        raise ValueError('Codec body absent from tracking XML')
                    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
                    if torso < 0:
                        raise ValueError('No torso body')
                    ee_names = ['left_ankle_roll_link', 'right_ankle_roll_link',
                                'left_wrist_yaw_link', 'right_wrist_yaw_link']
                    ee = [codec.body_names.index(n) for n in ee_names]
                    ref_data = mujoco.MjData(model)
                    measured_data = mujoco.MjData(model)
                    ref_bodies, ref_torso = [], []
                    for state in native:
                        ref_data.qpos[:7] = state[:7]
                        ref_data.qpos[qpos_ids] = state[13:42]
                        mujoco.mj_forward(model, ref_data)
                        ref_bodies.append(ref_data.xpos[body_ids].copy())
                        ref_torso.append(ref_data.xmat[torso].reshape(3, 3).copy())
                    ref_bodies, ref_torso = np.stack(ref_bodies), np.stack(ref_torso)
                    initial = {'root_pos': native[0, :3], 'root_quat_xyzw': native[0, [4, 5, 6, 3]],
                        'root_lin_vel': native[0, 7:10],
                        # MuJoCo floating-root angular qvel is body-local.
                        'root_ang_vel': quat_matrix(native[0, 3:7]).T @ native[0, 10:13],
                        'dof_pos': native[0, 13:42][order], 'dof_vel': native[0, 42:71][order]}
                    obs = env.reset(init_state_dict=initial)
                    env._gather_reference_state()
                    obs = env._update_observation_manager()
                    row['initial_root_pose_max_abs_error'] = float(np.max(np.abs(data.qpos[:7]-native[0, :7])))
                    # Official xyzw->wxyz reset converts through float32. This
                    # bound checks interface rounding, not a physical threshold.
                    np.testing.assert_allclose(data.qpos[:7], native[0, :7], atol=1e-7, rtol=0)
                    np.testing.assert_allclose(data.qpos[qpos_ids], native[0, 13:42], atol=1e-8, rtol=0)
                    frames, qposes, bodies, actions, errors, terms, sim_times = [], [], [], [], [], [], []
                    started = time.monotonic()
                    with torch.inference_mode():
                        for frame in range(len(native)):
                            tgt, action = agent.get_action(obs)
                            if not torch.isfinite(tgt).all() or not torch.isfinite(action).all():
                                raise FloatingPointError('Nonfinite external policy action')
                            obs = env.step(tgt, action)
                            # mj_step integrates qpos after computing derived
                            # xpos/xmat. Recompute ONLY measurement kinematics in
                            # separate storage, never the live integration state.
                            measured_data.qpos[:] = data.qpos
                            mujoco.mj_kinematics(model, measured_data)
                            actual = measured_data.xpos[body_ids].copy()
                            actual_r = measured_data.xmat[torso].reshape(3, 3)
                            delta = np.linalg.norm(actual-ref_bodies[frame], axis=-1)
                            pelvic = float(abs(data.qpos[2]-native[frame, 2]))
                            gravity = float(np.linalg.norm(actual_r.T @ np.array([0, 0, -1.]) -
                                ref_torso[frame].T @ np.array([0, 0, -1.])))
                            end = float(np.abs(actual[ee, 2]-ref_bodies[frame, ee, 2]).max())
                            if not np.isfinite(data.qpos).all() or not np.isfinite(actual).all():
                                raise FloatingPointError('Nonfinite actual physics state')
                            flags = np.array([pelvic > .3, gravity > .8, end > .3], dtype=bool)
                            frames.append(frame);qposes.append(data.qpos.copy());bodies.append(actual)
                            actions.append(action[0].cpu().numpy().copy());sim_times.append(float(data.time))
                            errors.append([float(delta.mean()), pelvic, gravity, end]);terms.append(flags)
                            if flags.any():
                                break
                    errors, terms = np.asarray(errors), np.asarray(terms)
                    complete = len(frames) == len(native)
                    passed = complete and not terms.any() and float(errors[:, 0].mean()) < .1
                    raw = episode/'actual_pre_reset.npz'
                    np.savez_compressed(raw, frame=np.asarray(frames), time=np.asarray(sim_times),
                        qpos=np.stack(qposes), body_pos=np.stack(bodies), actions=np.stack(actions),
                        errors=errors, termination_flags=terms, reference_body_pos=ref_bodies,
                        reference_native=native, body_names=np.asarray(codec.body_names))
                    row.update(status='AUXILIARY_SCREEN_COMPLETED', control_frames=len(frames),
                        full_clip_executed=complete, auxiliary_pass=bool(passed),
                        terminal_failure=bool(terms.any()), mean_mpjpe_m=float(errors[:, 0].mean()),
                        max_pelvis_z_error_m=float(errors[:, 1].max()),
                        max_torso_gravity_difference=float(errors[:, 2].max()),
                        max_end_effector_z_error_m=float(errors[:, 3].max()),
                        final_failure_terms=[n for n, flag in zip(['pelvis_z', 'torso_gravity', 'end_effector_z'], terms[-1]) if flag],
                        raw_sha256=sha(raw), elapsed_seconds=time.monotonic()-started,
                        geometry_xml_sha256=sha(xml), raw_path=str(raw))
                    save(episode/'receipt.json', row)
                    report['records'].append(row)
                    report.update(status='RUNNING', completed_episodes=len(report['records']))
                    save(a.output/'summary.json', report)
                    print(json.dumps({k: row[k] for k in ['index', 'seed', 'control_frames', 'reference_frames', 'auxiliary_pass', 'mean_mpjpe_m']}), flush=True)
                except BaseException as exc:
                    row.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
                    if hasattr(sim, 'mujoco_data'):
                        d = sim.mujoco_data
                        np.savez_compressed(episode/'failure_state_snapshot.npz', qpos=d.qpos.copy(),
                                            qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), time=np.asarray(d.time))
                    save(episode/'receipt.json', row)
                    report['records'].append(row)
                    raise
                finally:
                    sim.close()
        if any(sha(p) != h for p, h in bound.items()):
            raise ValueError('An immutable source or checkpoint changed')
        after = hashlib.sha256()
        for name, tensor in agent.policy.state_dict().items():
            after.update(name.encode());after.update(tensor.cpu().contiguous().numpy().tobytes())
        if after.hexdigest() != report['policy_parameter_digest_before']:
            raise ValueError('Frozen external policy changed')
        accepted = []
        for index in sorted(set(r['index'] for r in report['records'])):
            rows = [r for r in report['records'] if r['index'] == index]
            if len(rows) == len(a.seeds) and all(r['auxiliary_pass'] for r in rows):
                accepted.append(index)
        report.update(status='AUXILIARY_PHYSICAL_SCREEN_COMPLETED',
            all_seed_pass_indices=accepted, all_seed_pass_sequences=len(accepted),
            complete_episodes=sum(r['full_clip_executed'] for r in report['records']),
            passed_episodes=sum(r['auxiliary_pass'] for r in report['records']),
            actual_control_calls=sum(r['control_frames'] for r in report['records']),
            requested_control_calls=sum(r['reference_frames'] for r in report['records']),
            inputs_unchanged=True, policy_parameters_unchanged=True,
            interpretation='Only this external frozen tracker, seeds, geometry and complete horizons. '
                'Failure is not proof that a motion is intrinsically infeasible. No author GMT or curriculum claim.')
        save(a.output/'input_hashes_private.json', bound)
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(a.output/'summary.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['runtime-root', 'code-root', 'phygile-code', 'manifest', 'output']:
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds) or min(a.start, a.limit) < 0:
        p.error('Unique seeds and nonnegative selection required')
    run(a)
