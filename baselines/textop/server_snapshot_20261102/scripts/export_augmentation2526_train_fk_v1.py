"""DRAFT: Isaac FK for generated TRAIN clips, no dynamics or policy.

Requires a future independently accepted complete generation audit and
verified input adapter; no executable supervisor or bridge exists yet.
"""
import sys
if sys.flags.optimize:
    raise RuntimeError('optimized Python disables required checks')

import argparse
import hashlib
import os
import shutil
import subprocess
import time
import traceback
from step4_train_only_protocol import ROOT, RESERVE, LOG, owned, sha, read, atomic
from augmentation2526_integrity_fk_inputs_v1 import accepted_generated_fk_inputs

# Verify the dedicated supervisor is an ancestor and owns the common lock
# before importing or initializing Isaac.
import pathlib
pre = argparse.ArgumentParser(add_help=False); pre.add_argument('--request', required=True)
pre_args, _ = pre.parse_known_args()
pre_request = owned(pre_args.request)
from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
pid = os.getppid(); found = False
for _ in range(8):
    if pid <= 1: break
    proc = pathlib.Path('/proc') / str(pid)
    argv = (proc / 'cmdline').read_bytes().decode().split('\0')
    supervisor = str(ROOT / 'scripts/supervise_augmentation2526_train_fk_v1.py')
    if proc.stat().st_uid == 1013 and supervisor in argv and str(pre_request) in argv:
        found = True; break
    pid = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[1])
if not found: raise RuntimeError('dedicated FK supervisor required')
lock_path = owned(ROOT / 'receipts/full_library_capacity.lock'); lock_stat = lock_path.stat()
if not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(), pid,
                               os.major(lock_stat.st_dev), os.minor(lock_stat.st_dev), lock_stat.st_ino):
    raise RuntimeError('FK supervisor does not hold the shared lock')
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument('--request', required=True)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
sp = owned(a.request); s = read(sp)
assert os.getuid() == 1013 and a.headless and a.device == 'cuda:1'
assert 'CUDA_VISIBLE_DEVICES' not in os.environ and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
assert set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
assert sha(__file__) == s['executor_sha256']
for n, h in s['bound_sources'].items(): assert sha(owned(ROOT / n)) == h
assert s['fk_authorized'] is True and s['role'] in ('author', 'document', 'library')
allowed = {'author': 'user_augmentation2526_fk_author_v1',
           'document': 'user_augmentation2526_fk_document_v1',
           'library': 'user_augmentation2526_fk_library_v1'}
assert s['run'] == allowed[s['role']] and s['output_root'] == str(ROOT / 'step34_20261029')
metadata = accepted_generated_fk_inputs(s['audit_binding'])
binding = metadata['binding']
assert binding == s['input_binding']
all_clips = metadata['plan']['inputs']
assert len(all_clips) == 2526 and metadata['plan']['output_npz_upper_bytes'] == 5764264784
clips = all_clips if s['role'] == 'library' else all_clips[:1]
assert s['output_upper_bytes'] >= sum(x['frames']*2152+8192 for x in clips) + 128*1024**2
assert s['future_state_upper_bytes'] == 6*128*1024**2
spec = read(ROOT / 'configs/tracker-env-spec.json')
assert sha(spec['robot_urdf']) == spec['robot_urdf_sha256'] == s['urdf_sha256'] == metadata['original_urdf_sha256']
parent = owned(s['output_root']); out = parent / s['run']; out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic(); app = sim = None
rec = dict(status='STARTING_AUGMENTATION_TRAIN_REFERENCE_FK', run=s['run'], role=s['role'],
    source_sha256=sha(__file__), request_sha256=sha(sp), input_binding=binding,
    training_data_mode='generated_train_only', stage3_status='GENERATED_FK_IN_PROGRESS',
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
    assert shutil.disk_usage(out).free >= RESERVE + LOG + s['output_upper_bytes'] + s['future_state_upper_bytes']
    app = AppLauncher(a).app
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from textop_tracker.robots.g1 import G1_CYLINDER_CFG
    torch.set_num_threads(4); torch.manual_seed(20261103)
    physical_uuid = subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=uuid',
        '--format=csv,noheader,nounits'], text=True).strip().lower()
    runtime_uuid = getattr(torch.cuda.get_device_properties(torch.device(a.device)), 'uuid', None)
    assert runtime_uuid is not None and str(runtime_uuid).lower() == physical_uuid
    rec['physical_gpu_uuid'] = physical_uuid
    torch.cuda.set_per_process_memory_fraction(.65, torch.device(a.device))
    assert torch.cuda.mem_get_info(torch.device(a.device))[0] >= 20*1024**3
    cfg = G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    cfg.spawn.asset_path = spec['robot_urdf']; cfg.spawn.usd_dir = str(out/'usd'); cfg.spawn.usd_file_name = 'g1.usd'

    @configclass
    class Scene(InteractiveSceneCfg):
        robot: ArticulationCfg = cfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=a.device))
    scene = InteractiveScene(Scene(num_envs=128, env_spacing=2.5)); sim.reset(); robot = scene['robot']
    assert list(robot.joint_names) == metadata['original_joint_names']
    assert list(robot.body_names) == metadata['original_body_names']
    rec.update(joint_names=list(robot.joint_names), body_names=list(robot.body_names), fps=50,
        quaternion_convention='WXYZ', asset_equivalence_to_author_not_established=True,
        velocity_method='numpy.gradient .02; original adjacent quaternion world angular velocity',
        numeric_source_sha256=s['bound_sources']['scripts/export_tracker_references.py'])
    save(); manifest = out / 'references.private.jsonl'; output_bytes = 0
    with manifest.open('x', encoding='utf-8') as journal:
        os.chmod(manifest, 0o600)
        for index, entry in enumerate(clips):
            remaining = sum((x['control_intervals']+1)*2152+8192 for x in clips[index:])
            assert shutil.disk_usage(out).free >= RESERVE + LOG + remaining + 128*1024**2 + s['future_state_upper_bytes']
            path = owned(entry['source_file']); before = path.stat()
            assert sha(path) == entry['source_sha256']
            with np.load(path, allow_pickle=False) as data:
                full = np.array(data['qpos_29'], dtype=np.float32, copy=True)
            assert full.shape == (entry['frames'], 36) and np.isfinite(full).all()
            assert hashlib.sha256(full.tobytes()).hexdigest() == entry['qpos_sha256']
            q = full; assert len(q) == entry['control_intervals']+1
            assert np.max(np.abs(np.sum(q[:, 3:7].astype(np.float64)**2, axis=1)-1)) < 2e-3
            order = [entry['joint_names'].index(n) for n in robot.joint_names]
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
            key = 'generated_TRAIN|' + str(entry['index']) + '|' + entry['source_sha256']
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
            rec['exported_clips'] += 1; rec['generated_training_clips'] += 1; rec['exported_frames'] += len(q); rec['control_intervals'] += len(q)-1
            save()
    assert rec['exported_clips'] == len(clips)
    if s['role'] == 'library': assert rec['control_intervals'] == 2666420 and rec['exported_frames'] == 2668946
    assert accepted_generated_fk_inputs(s['audit_binding'])['binding'] == binding
    rec.update(status='AUGMENTATION_TRAIN_ISAAC_REFERENCES_FINISHED', output_writers_closed=True,
        private_manifest=dict(path=str(manifest), bytes=manifest.stat().st_size, sha256=sha(manifest)),
        output_npz_bytes=output_bytes, source_bytes_unchanged=True,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(a.device)))
    save()
except BaseException as error:
    rec.update(status='AUGMENTATION_TRAIN_REFERENCE_FAILED_PARTIAL_RETAINED', error=repr(error), traceback=traceback.format_exc()); save(); raise
finally:
    if app is not None:
        from bounded_headless_close import close
        close(app, rec, save, out, simulation=sim, successful_statuses=('AUGMENTATION_TRAIN_ISAAC_REFERENCES_FINISHED',))
