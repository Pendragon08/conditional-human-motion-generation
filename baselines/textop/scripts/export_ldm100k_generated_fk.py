"""Actual Isaac FK of genuine generated futures; zero dynamics and zero training.

Keeps every mode/seed and raw joint value. This adapts the previously executed
reference FK mapping to the completed generation's own explicit provenance.
It does not fabricate an original-data preprocessing/filtering receipt.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
import traceback
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
ROOT = pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
HOME = pathlib.Path('/home/ouyanghao')


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def owned(path, root):
    path = pathlib.Path(path)
    assert path.is_absolute() and path.resolve() == path and (root == path or root in path.parents)
    assert path.stat().st_uid == 1013 and not path.is_symlink()
    for parent in path.parents:
        assert not parent.is_symlink()
        if parent == root:
            break
    return path


def read(path):
    return json.loads(pathlib.Path(path).read_bytes())


assert os.getuid() == 1013 and a.headless and a.device == 'cuda:1'
assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
request = owned(a.request, ROOT)
s = read(request)
assert sha(__file__) == s['executor_sha256']
assert sha(ROOT / 'configs/tracker-env-spec.json') == s['tracker_spec_sha256']
spec = read(ROOT / 'configs/tracker-env-spec.json')
assert sha(spec['robot_urdf']) == spec['robot_urdf_sha256'] == s['urdf_sha256']
assert sha(ROOT / 'scripts/export_tracker_references.py') == s['original_fk_source_sha256']
assert sha(ROOT / 'scripts/bounded_headless_close.py') == s['close_helper_sha256']
summary_path = owned(s['summary_receipt'], HOME)
assert sha(summary_path) == s['summary_receipt_sha256']
summary = read(summary_path)
assert summary['status'] == 'USER_LDM100K_DIAGNOSTIC32_CPU_SUMMARY_FINISHED' and summary['output_writers_closed']
item = summary['private_artifacts']['physical_inputs.private.json']
manifest_path = owned(item['path'], HOME)
assert sha(manifest_path) == item['sha256'] and manifest_path.stat().st_size == item['bytes']
records = read(manifest_path)
assert len(records) == summary['actual_trials'] == 1880
assert len({(x['segment_id'], x['seed'], x['mode']) for x in records}) == len(records)
assert all(x['source_kind'] == 'REAL_LDM100000_GENERATED_SCORED_FUTURE' and x['physical_filter_passed'] is False for x in records)
parent = owned(s['output_root'], HOME)
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic()
app = sim = None
rec = dict(status='STARTING_USER_LDM100K_GENERATED_FK', run=s['run'], source_sha256=sha(__file__),
    request_sha256=sha(request), input_summary_receipt_sha256=sha(summary_path),
    input_manifest_sha256=item['sha256'], declared_trials=len(records), exported_trials=0,
    exported_frames=0, no_transition_trials=0, optimizer_updates=0, neural_model_loads=0,
    physics_steps=0, physical_filter_executed=False, physical_quality_accepted=False,
    output_writers_closed=False, quaternion_convention='WXYZ', fps=50)


def save():
    rec['elapsed_seconds'] = time.monotonic() - started
    temp = out / 'receipt.partial'
    with temp.open('w') as f:
        json.dump(rec, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(out / 'receipt.json')


def angular_velocity(q, dt):
    prev, nxt = q[:-1], q[1:]
    w = np.sum(prev * nxt, axis=-1)
    v = prev[..., :1] * nxt[..., 1:] - nxt[..., :1] * prev[..., 1:] + np.cross(prev[..., 1:], nxt[..., 1:])
    sign = np.where(w < 0, -1., 1.)
    w, v = w * sign, v * sign[..., None]
    norm = np.linalg.norm(v, axis=-1)
    adjacent = v * (2 * np.arctan2(norm, w) / np.maximum(norm, 1e-12))[..., None] / dt
    return np.concatenate([adjacent[:1], (adjacent[:-1] + adjacent[1:]) * .5, adjacent[-1:]], axis=0)


save()
try:
    assert shutil.disk_usage(out).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + s['output_upper_bytes']
    app = AppLauncher(a).app
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from textop_tracker.robots.g1 import G1_CYLINDER_CFG
    torch.set_num_threads(4)
    torch.manual_seed(20261029)
    torch.cuda.set_per_process_memory_fraction(.65, torch.device(a.device))
    assert torch.cuda.mem_get_info(torch.device(a.device))[0] >= 20 * 1024**3
    robot_cfg = G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    robot_cfg.spawn.asset_path = spec['robot_urdf']
    robot_cfg.spawn.usd_dir = str(out / 'usd')
    robot_cfg.spawn.usd_file_name = 'g1.usd'

    @configclass
    class Scene(InteractiveSceneCfg):
        robot: ArticulationCfg = robot_cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=a.device))
    scene = InteractiveScene(Scene(num_envs=128, env_spacing=2.5))
    sim.reset()
    robot = scene['robot']
    assert len(robot.joint_names) == 29 and len(robot.body_names) == 30
    rec.update(joint_names=list(robot.joint_names), body_names=list(robot.body_names),
               urdf_sha256=sha(spec['robot_urdf']), asset_equivalence_to_author_not_established=True,
               numeric_method_source_sha256=s['original_fk_source_sha256'])
    save()
    manifest = out / 'fk_manifest.private.jsonl'
    output_bytes = 0
    with manifest.open('x', encoding='utf-8') as f:
        os.chmod(manifest, 0o600)
        for index, entry in enumerate(records):
            # Original quaternion differentiation promotes angular velocities
            # to float64:358 float32 +90 float64 values =2152B/frame.
            remaining = sum((x['frames'] * 2152 + 8192) for x in records[index:] if x['transition_applicable'])
            assert shutil.disk_usage(out).free >= s['reserve_bytes'] + s['log_allowance_bytes'] + remaining
            source = owned(entry['generated_npz'], ROOT)
            before = source.stat()
            assert before.st_size == entry['generated_npz_bytes'] and sha(source) == entry['generated_npz_sha256']
            with np.load(source, allow_pickle=False) as data:
                q = np.array(data[entry['qpos_key']], dtype=np.float32, copy=True)
            assert list(q.shape) == entry['qpos_shape'] == [entry['frames'], 36] and np.isfinite(q).all()
            assert hashlib.sha256(q.tobytes()).hexdigest() == entry['qpos_sha256']
            assert len(entry['joint_names']) == len(set(entry['joint_names'])) == 29
            assert np.max(np.abs(np.sum(q[:, 3:7].astype(np.float64)**2, axis=1) - 1)) < 2e-3
            if not entry['transition_applicable']:
                assert len(q) == 1
                f.write(json.dumps(dict(entry, status='NO_TRANSITION_SINGLE_FRAME_RETAINED', physical_failure_claimed=False)) + '\n')
                rec['no_transition_trials'] += 1
                continue
            assert len(q) >= 2
            order = [entry['joint_names'].index(n) for n in robot.joint_names]
            jp = q[:, 7:][:, order]
            positions, quaternions = [], []
            for start in range(0, len(q), 128):
                n = min(128, len(q) - start)
                indices = np.minimum(np.arange(128) + start, len(q) - 1)
                root = robot.data.default_root_state.clone()
                root[:, :3] = torch.as_tensor(q[indices, :3], device=a.device) + scene.env_origins
                root[:, 3:7] = torch.as_tensor(q[indices, 3:7], device=a.device)
                root[:, 7:] = 0
                joints = torch.as_tensor(jp[indices], device=a.device)
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
                sim.forward()
                scene.update(.005)
                assert torch.allclose(robot.data.joint_pos, joints, atol=1e-5)
                positions.append((robot.data.body_pos_w - scene.env_origins[:, None, :])[:n].cpu().numpy())
                quaternions.append(robot.data.body_quat_w[:n].cpu().numpy())
            bp, bq = np.concatenate(positions), np.concatenate(quaternions)
            assert np.isfinite(bp).all() and np.isfinite(bq).all()
            assert np.max(np.abs(np.linalg.norm(bq, axis=-1) - 1)) < 1e-4
            key = entry['segment_id'] + '|' + str(entry['seed']) + '|' + entry['mode']
            dest = out / (hashlib.sha256(key.encode()).hexdigest() + '.private.npz')
            arrays = dict(fps=np.array([50]), joint_pos=jp, joint_vel=np.gradient(jp, .02, axis=0),
                body_pos_w=bp, body_quat_w=bq, body_lin_vel_w=np.gradient(bp, .02, axis=0),
                body_ang_vel_w=angular_velocity(bq, .02))
            assert all(np.isfinite(v).all() for v in arrays.values())
            with dest.open('xb') as z:
                os.chmod(dest, 0o600)
                np.savez(z, **arrays)
                z.flush()
                os.fsync(z.fileno())
            assert dest.stat().st_size <= len(q) * 2152 + 8192
            assert sha(source) == entry['generated_npz_sha256'] and source.stat().st_mtime_ns == before.st_mtime_ns
            record = dict(entry, status='REAL_GENERATED_ISAAC_FK_EXPORTED', file=str(dest), sha256=sha(dest), bytes=dest.stat().st_size,
                          numeric_array_hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()},
                          robot_joint_names=list(robot.joint_names), robot_body_names=list(robot.body_names),
                          no_joint_clamp_applied=True, dynamics_executed=False)
            f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())
            output_bytes += dest.stat().st_size
            rec['exported_trials'] += 1
            rec['exported_frames'] += len(q)
            save()
            if rec['exported_trials'] % 100 == 0:
                print(json.dumps({k: rec[k] for k in ['exported_trials', 'declared_trials', 'exported_frames', 'elapsed_seconds']}), flush=True)
    assert rec['exported_trials'] + rec['no_transition_trials'] == 1880
    assert sha(manifest_path) == item['sha256'] and sha(summary_path) == s['summary_receipt_sha256']
    assert sha(spec['robot_urdf']) == s['urdf_sha256']
    rec.update(status='USER_LDM100K_GENERATED_ISAAC_FK_FINISHED', output_writers_closed=True,
        private_manifest=dict(path=str(manifest), bytes=manifest.stat().st_size, sha256=sha(manifest)),
        private_fk_npz_bytes=output_bytes, generated_source_bytes_unchanged=True,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(a.device)))
    save()
except BaseException as error:
    rec.update(status='USER_LDM100K_GENERATED_FK_FAILED_PARTIAL_RETAINED', error=repr(error), traceback=traceback.format_exc())
    save()
    raise
finally:
    if app is not None:
        from bounded_headless_close import close
        close(app, rec, save, out, simulation=sim, successful_statuses=('USER_LDM100K_GENERATED_ISAAC_FK_FINISHED',))
