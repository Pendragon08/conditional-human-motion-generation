"""Diagnostic tracking of clock-gated 262D pairs using a frozen pilot policy.

Private trajectories stay in --output. This never promotes data into an author
difficulty level, changes the paired dataset, trains a policy, or claims paper
physical filtering. Full failures and first-termination censoring are retained.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def resample_poses(native, source_fps, target_fps):
    """Interpolate poses at fixed physical times; never stretch or extrapolate.

    The final source endpoint need not fall on the control grid. Record the
    omitted <1-control-tick tail instead of silently changing playback speed.
    """
    if native.ndim != 2 or native.shape[1] != 71 or len(native) < 2:
        raise ValueError('Expected at least two native 71D frames')
    if not np.isfinite(native).all() or not all(np.isfinite(f) and f > 0 for f in (source_fps, target_fps)):
        raise ValueError('Nonfinite pose or invalid clock')
    times = np.arange(len(native), dtype=np.float64)/source_fps
    count = int(np.floor(times[-1]*target_fps+1e-9))+1
    query = np.arange(count, dtype=np.float64)/target_fps
    if count < 2 or query[-1] > times[-1]+1e-10:
        raise ValueError('Control grid invalid or extrapolating')
    pos = np.stack([np.interp(query, times, native[:, i]) for i in range(3)], -1)
    joints = np.stack([np.interp(query, times, native[:, i]) for i in range(13, 42)], -1)
    rotations = Rotation.from_quat(native[:, [4, 5, 6, 3]])
    quat = Slerp(times, rotations)(query).as_quat()[:, [3, 0, 1, 2]]
    return pos, quat, joints, {'source_fps': source_fps, 'control_fps': target_fps,
        'source_frames': len(native), 'control_frames': count,
        'source_last_sample_s': float(times[-1]), 'control_last_sample_s': float(query[-1]),
        'omitted_tail_s': float(max(0, times[-1]-query[-1])),
        'method': 'linear root translation and joints; shortest-arc root quaternion SLERP',
        'time_stretched': False, 'extrapolation': False,
        'velocity': 'forward difference at control rate; final difference repeated'}


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'STARTING', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'evaluation_type': 'simulation_only', 'trained_generator': False, 'optimizer_updates': 0,
        'paper_physical_filter_reproduced': False, 'curriculum_labels_created': 0,
        'source_dataset_modified': False, 'policy_role': 'undertrained 2500-PPO-iteration pilot diagnostic',
        'contact_profile': 'official Unitree Mjlab geometry/friction', 'records': []}
    save(a.output/'summary.json', report)
    observed = {}
    def bind(path):
        path = Path(path).resolve(strict=True)
        observed[str(path)] = sha(path)
        return path
    try:
        manifest = json.loads(bind(a.manifest).read_text(encoding='utf-8'))
        if manifest['sample_clock_gate']['status'] != 'NUMERIC_CLOCK_CONSISTENCY_ONLY':
            raise ValueError('Clock-gated dataset required')
        entries = [(split, entry) for split, rows in manifest['splits'].items() for entry in rows]
        if a.limit:
            entries = entries[:a.limit]
        if not entries or len({(e['motion_id'], e['interval']) for _, e in entries}) != len(entries):
            raise ValueError('Empty or duplicated selection')
        report.update(selected_sequences=len(entries), source_manifest_sha256=sha(a.manifest),
                      partial_selection=bool(a.limit), selection_order='manifest split and row order')
        sys.path.insert(0, str(a.phygile_code/'reimplementation'))
        from codec262 import G1Codec, derivative, matrix_quat, matrix_rotvec, quat_matrix
        skeleton = bind(a.phygile_code/'configs/g1_codec_skeleton.json')
        bind(a.phygile_code/'reimplementation/codec262.py')
        bind(__file__)
        codec = G1Codec(skeleton)
        contract = json.loads(bind(a.assets/'contract.json').read_text(encoding='utf-8'))
        bind(a.assets/'encoder_actor_critic.py')
        checkpoint = bind(a.checkpoint)
        report['checkpoint_sha256'] = sha(checkpoint)
        rate = 1/float(contract['control_dt'])
        if tuple(contract['joint_names']) != codec.joint_names or abs(rate-50) > 1e-9:
            raise ValueError('Unexpected joint/control contract')
        prepared = a.output/'prepared_private'
        prepared.mkdir()
        for name in ['contract.json', 'encoder_actor_critic.py']:
            shutil.copyfile(a.assets/name, prepared/name)
        for index, (split, e) in enumerate(entries):
            if not e['sample_clock_gate']['eligible']:
                raise ValueError('Ineligible source')
            feature = bind(e['features']); native_file = bind(e['native']); metadata_file = bind(e['codec_metadata'])
            if sha(feature) != e['sha256'] or sha(native_file) != e['native_sha256']:
                raise ValueError('Dataset integrity mismatch')
            x = np.load(feature, allow_pickle=False)
            metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
            decoded, _, codec_report = codec.decode(x, metadata, project=False, restore_world=True)
            with np.load(native_file, allow_pickle=False) as source:
                native = source['native']
                if len(native) != e['frames'] or not np.all(np.diff(source['reference_raw_frames']) == e['reference_raw_stride']):
                    raise ValueError('Native shape/frame identity mismatch')
            pdelta = float(np.max(np.abs(decoded[:, :3]-native[:, :3])))
            qdelta = float(np.max(np.abs(decoded[:, 13:42]-native[:, 13:42])))
            rdelta = float(np.max(np.abs(quat_matrix(decoded[:, 3:7])-quat_matrix(native[:, 3:7]))))
            if max(pdelta, qdelta, rdelta) > 1e-5:
                raise ValueError('262D decode and stored native poses diverged')
            pos, quat, joints, clock = resample_poses(decoded, float(metadata['fps']), rate)
            positions, rotations = codec.fk(pos, quat, joints)
            names = contract['body_names']
            bp = np.stack([positions[n] for n in names], 1)
            br = np.stack([rotations[n] for n in names], 1)
            angular = matrix_rotvec(br[1:] @ br[:-1].transpose(0, 1, 3, 2))*rate
            motion = prepared/f'test_{index:03d}.npz'
            np.savez_compressed(motion, fps=np.array([rate], np.float32),
                joint_names=np.asarray(codec.joint_names), body_names=np.asarray(names),
                joint_pos=joints.astype(np.float32), joint_vel=derivative(joints, rate).astype(np.float32),
                body_pos_w=bp.astype(np.float32), body_quat_w=matrix_quat(br).astype(np.float32),
                body_lin_vel_w=derivative(bp, rate).astype(np.float32),
                body_ang_vel_w=np.concatenate([angular, angular[-1:]]).astype(np.float32))
            dest = a.output/f'episode_{index:03d}'
            command = [sys.executable, str(a.unitree_scripts/'mjlab_evaluate.py'),
                '--unitree-checkout', str(a.unitree_checkout), '--assets', str(prepared),
                '--checkpoint', str(checkpoint), '--output', str(dest), '--clip', str(index),
                '--seed', str(a.seed), '--contact-profile', 'official', '--device', 'cpu']
            row = {'index': index, 'motion_id': e['motion_id'], 'interval': e['interval'],
                'split': split, 'source_group': e['source_group'], 'mirrored': e['mirrored'],
                'features_sha256': sha(feature), 'native_sha256': sha(native_file),
                'codec_metadata_sha256': sha(metadata_file), 'decoded_reference_sha256': sha(motion),
                'decode_max_abs': {'root_position_m': pdelta, 'joint_rad': qdelta, 'rotation_matrix': rdelta},
                'codec_report': codec_report, 'clock': clock, 'seed': a.seed, 'command': command}
            save(a.output/f'episode_{index:03d}_request.json', row)
            env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       WANDB_MODE='disabled', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
            with (a.output/f'episode_{index:03d}_stdout.txt').open('w', encoding='utf-8') as out, \
                 (a.output/f'episode_{index:03d}_stderr.txt').open('w', encoding='utf-8') as err:
                result = subprocess.run(command, stdout=out, stderr=err, env=env, timeout=600)
            row['exit_code'] = result.returncode
            if (dest/'receipt.json').exists():
                receipt = json.loads((dest/'receipt.json').read_text(encoding='utf-8'))
                row['receipt_sha256'] = sha(dest/'receipt.json')
                row['receipt'] = {k: receipt.get(k) for k in ['status', 'checkpoint_sha256', 'motion_sha256',
                    'frames_evaluated', 'motion_frames', 'full_clip_success', 'terminated', 'timeout',
                    'metrics', 'control_dt', 'physics_dt', 'executed_sources_unchanged']}
            report['records'].append(row)
            report.update(status='RUNNING', completed_sequences=len(report['records']))
            save(a.output/'summary.json', report)
            if result.returncode:
                raise RuntimeError(f'Physical evaluator failed at {index}; original files/logs retained')
            if (receipt['status'] != 'FIRST_EPISODE_EVALUATED' or not receipt['executed_sources_unchanged']
                    or receipt['checkpoint_sha256'] != sha(checkpoint) or receipt['motion_sha256'] != sha(motion)):
                raise ValueError('Evaluation receipt/integrity invalid')
            with np.load(dest/'raw_pre_reset.npz', allow_pickle=False) as raw:
                if not all(np.isfinite(raw[k]).all() for k in raw.files):
                    raise ValueError('Nonfinite full recorded episode')
                if not np.array_equal(raw['frame'], np.arange(receipt['frames_evaluated'])):
                    raise ValueError('Dropped reference frames')
            row['raw_sha256'] = sha(dest/'raw_pre_reset.npz')
            print(json.dumps({'completed': index+1, 'of': len(entries), 'motion': e['motion_id'],
                  'success': receipt['full_clip_success'], 'frames': receipt['frames_evaluated']}), flush=True)
        if any(sha(p) != h for p, h in observed.items()):
            raise ValueError('Input modified during evaluation')
        save(a.output/'input_hashes.json', observed)
        rows = report['records']
        report.update(status='DIAGNOSTIC_SCREEN_COMPLETED', inputs_unchanged=True,
            full_clip_successes=sum(r['receipt']['full_clip_success'] for r in rows),
            observed_control_frames=sum(r['receipt']['frames_evaluated'] for r in rows),
            requested_control_frames=sum(r['clock']['control_frames'] for r in rows),
            unique_sources=len({r['source_group'] for r in rows}),
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            interpretation='One deterministic first episode per sequence. Errors censored at first termination. '
            'Mirrors and overlapping captions are correlated; not independent trials. No data removed or approved for paper training.')
        save(a.output/'summary.json', report)
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        save(a.output/'summary.json', report)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['manifest', 'phygile-code', 'unitree-scripts', 'unitree-checkout', 'assets', 'checkpoint', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.limit < 0:
        p.error('limit must be nonnegative')
    run(args)
