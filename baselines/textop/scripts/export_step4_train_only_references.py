"""New TRAIN-only Isaac FK, no dynamics, no generator, no teacher.

Author and fresh document run the identical first real TRAIN slice. A separate
authorized library run converts every original TRAIN slice, with no VAL inputs.
"""
import argparse
import hashlib
import os
import shutil
import time
import traceback
from isaaclab.app import AppLauncher
from step4_train_only_protocol import ROOT, RESERVE, LOG, owned, sha, read, atomic, select

p = argparse.ArgumentParser(); p.add_argument('--request', required=True)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
sp = owned(a.request); s = read(sp)
assert os.getuid() == 1013 and a.headless and a.device == 'cuda:1'
assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
assert sha(__file__) == s['executor_sha256']
for n, h in s['bound_sources'].items(): assert sha(owned(ROOT / n)) == h
clips, sources, names, binding = select(s['role'])
assert binding == s['input_binding']
spec = read(ROOT / 'configs/tracker-env-spec.json')
assert sha(spec['robot_urdf']) == spec['robot_urdf_sha256'] == s['urdf_sha256']
parent = owned(s['output_root']); out = parent / s['run']; out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic(); app = sim = None
rec = dict(status='STARTING_STAGE4_ORIGINAL_TRAIN_REFERENCE_FK', run=s['run'], role=s['role'],
    source_sha256=sha(__file__), request_sha256=sha(sp), input_binding=binding,
    training_data_mode='original_train_only', stage3_status='POSTPONED_BY_USER',
    target_clips=len(clips), exported_clips=0, exported_frames=0, control_intervals=0,
    generated_training_clips=0, original_VAL_motions_in_training=0, neural_model_loads=0,
    optimizer_updates=0, physics_steps=0, output_writers_closed=False)


def save():
    rec['elapsed_seconds'] = time.monotonic() - started
    atomic(out / 'receipt.json', rec)


def angular(q):
    prev, nxt = q[:-1], q[1:]
    w = np.sum(prev * nxt, axis=-1)
    v = prev[..., :1]*nxt[..., 1:] - nxt[..., :1]*prev[..., 1:] + np.cross(prev[..., 1:], nxt[..., 1:])
    sign = np.where(w < 0, -1., 1.); w, v = w*sign, v*sign[..., None]
    norm = np.linalg.norm(v, axis=-1)
    adjacent = v*(2*np.arctan2(norm, w)/np.maximum(norm, 1e-12))[..., None]/.02
    return np.concatenate([adjacent[:1], (adjacent[:-1]+adjacent[1:])*.5, adjacent[-1:]])


save()
try:
    assert shutil.disk_usage(out).free >= RESERVE + LOG + s['output_upper_bytes']
    app = AppLauncher(a).app
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from textop_tracker.robots.g1 import G1_CYLINDER_CFG
    torch.set_num_threads(4); torch.manual_seed(20261103)
    torch.cuda.set_per_process_memory_fraction(.65, torch.device(a.device))
    assert torch.cuda.mem_get_info(torch.device(a.device))[0] >= 20*1024**3
    cfg = G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    cfg.spawn.asset_path = spec['robot_urdf']; cfg.spawn.usd_dir = str(out/'usd'); cfg.spawn.usd_file_name = 'g1.usd'

    @configclass
    class Scene(InteractiveSceneCfg):
        robot: ArticulationCfg = cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=a.device))
    scene = InteractiveScene(Scene(num_envs=128, env_spacing=2.5)); sim.reset(); robot = scene['robot']
    assert len(robot.joint_names) == 29 and len(robot.body_names) == 30
    rec.update(joint_names=list(robot.joint_names), body_names=list(robot.body_names), fps=50,
        quaternion_convention='WXYZ', asset_equivalence_to_author_not_established=True,
        velocity_method='numpy.gradient .02; original adjacent quaternion world angular velocity',
        numeric_source_sha256=s['bound_sources']['scripts/export_tracker_references.py'])
    save(); manifest = out / 'references.private.jsonl'; output_bytes = 0
    with manifest.open('x', encoding='utf-8') as journal:
        os.chmod(manifest, 0o600)
        for index, entry in enumerate(clips):
            remaining = sum((x['control_intervals']+1)*2152+8192 for x in clips[index:])
            assert shutil.disk_usage(out).free >= RESERVE + LOG + remaining
            path = owned(entry['source_path']); before = path.stat()
            assert sha(path) == entry['source_sha256']
            with np.load(path, allow_pickle=False) as data:
                full = np.array(data['qpos_29'], dtype=np.float32, copy=True)
            assert full.shape == (sources[entry['source_family']]['frames50'], 36) and np.isfinite(full).all()
            first, last = entry['first_pose_index'], entry['last_pose_index_inclusive']
            q = full[first:last+1]; assert len(q) == entry['control_intervals']+1
            assert np.max(np.abs(np.sum(q[:, 3:7].astype(np.float64)**2, axis=1)-1)) < 2e-3
            order = [names[entry['source_family']].index(n) for n in robot.joint_names]
            jp = q[:, 7:][:, order]; positions, quaternions = [], []
            for start in range(0, len(q), 128):
                count = min(128, len(q)-start); indices = np.minimum(np.arange(128)+start, len(q)-1)
                root = robot.data.default_root_state.clone()
                root[:, :3] = torch.as_tensor(q[indices, :3], device=a.device)+scene.env_origins
                root[:, 3:7] = torch.as_tensor(q[indices, 3:7], device=a.device); root[:, 7:] = 0
                joints = torch.as_tensor(jp[indices], device=a.device)
                robot.write_root_state_to_sim(root); robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
                sim.forward(); scene.update(.005)
                assert torch.allclose(robot.data.joint_pos, joints, atol=1e-5)
                positions.append((robot.data.body_pos_w-scene.env_origins[:, None, :])[:count].cpu().numpy())
                quaternions.append(robot.data.body_quat_w[:count].cpu().numpy())
            bp, bq = np.concatenate(positions), np.concatenate(quaternions)
            arrays = dict(fps=np.array([50]), joint_pos=jp, joint_vel=np.gradient(jp, .02, axis=0),
                body_pos_w=bp, body_quat_w=bq, body_lin_vel_w=np.gradient(bp, .02, axis=0), body_ang_vel_w=angular(bq))
            assert all(np.isfinite(v).all() for v in arrays.values())
            assert np.max(np.abs(np.linalg.norm(bq, axis=-1)-1)) < 1e-4
            key = entry['source_family']+'|'+str(first)+'|'+str(last)
            target = out / (hashlib.sha256(key.encode()).hexdigest()+'.private.npz')
            with target.open('xb') as f:
                os.chmod(target, 0o600); np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
            assert target.stat().st_size <= len(q)*2152+8192
            assert sha(path) == entry['source_sha256'] and path.stat().st_mtime_ns == before.st_mtime_ns
            hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}
            row = dict(entry, file=str(target), bytes=target.stat().st_size, sha256=sha(target), frames=len(q),
                array_sha256=hashes, array_shapes={k: list(v.shape) for k, v in arrays.items()},
                array_dtypes={k: str(v.dtype) for k, v in arrays.items()}, no_joint_clamp_applied=True)
            import json
            journal.write(json.dumps(row, ensure_ascii=False)+'\n'); journal.flush(); os.fsync(journal.fileno())
            output_bytes += target.stat().st_size
            rec['exported_clips'] += 1; rec['exported_frames'] += len(q); rec['control_intervals'] += len(q)-1
            save()
    assert rec['exported_clips'] == len(clips)
    if s['role'] == 'library': assert rec['control_intervals'] == 2835665
    assert select(s['role'])[3] == binding
    rec.update(status='STAGE4_ORIGINAL_TRAIN_ISAAC_REFERENCES_FINISHED', output_writers_closed=True,
        private_manifest=dict(path=str(manifest), bytes=manifest.stat().st_size, sha256=sha(manifest)),
        output_npz_bytes=output_bytes, source_bytes_unchanged=True,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(a.device)))
    save()
except BaseException as error:
    rec.update(status='STAGE4_ORIGINAL_TRAIN_REFERENCE_FAILED_PARTIAL_RETAINED', error=repr(error), traceback=traceback.format_exc()); save(); raise
finally:
    if app is not None:
        from bounded_headless_close import close
        close(app, rec, save, out, simulation=sim, successful_statuses=('STAGE4_ORIGINAL_TRAIN_ISAAC_REFERENCES_FINISHED',))
