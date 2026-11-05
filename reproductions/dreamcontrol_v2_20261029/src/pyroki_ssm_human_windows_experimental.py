"""Retarget one marker-aligned SSM human witness using isolated CPU PyRoki.

Experimental old DreamControl objectives and assumed ground. These windows
are not a training dataset, not stitched, and not the unpublished v2 method.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
VARIANT = ROOT / 'pyroki_grab_temporal_20261030'
sys.path.insert(0, str(VARIANT))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof, create_conn_tree, solve_retargeting, foot_detect


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    if any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('CPU required')
    source = ROOT / 'humanml_mirror_time_witness_v2_20261030/000008_joint_witness.npz'
    expected = '8354f592da84aa9e62b87f4ab467465cd8a43120e39d5c8ae346966a8883a0b7'
    receipt = json.loads((VARIANT / 'report.json').read_text())
    helper = VARIANT / 'retarget_helpers/_utils.py'
    urdf_path = PROJECT / 'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    if sha(source) != expected or sha(helper) != receipt['helper_sha256'] or sha(urdf_path) != receipt['urdf_sha256']:
        raise ValueError('source, helper or URDF hash differs from prior audit')
    with np.load(source, allow_pickle=False) as d:
        converted = d['original_joints_humanml'].astype(np.float32)
        source_frames = d['source_frames'].copy()
        times = d['source_times'].copy()
    if converted.shape != (43, 22, 3) or not np.isfinite(converted).all() or len(times) != 43:
        raise ValueError('unexpected SSM witness')
    # Inverse of the pinned official non-HumanAct12 SMPL-to-HumanML axis map.
    human = np.stack((-converted[..., 0], converted[..., 2], converted[..., 1]), axis=-1)
    if not np.all(np.diff(source_frames) == 6) or not np.all(np.diff(times) > 0):
        raise ValueError('invalid source timeline')
    fps = float(1 / np.median(np.diff(times)))
    if abs(fps - 20.00000762939453) > 1e-6:
        raise ValueError('reference-derived FPS changed')
    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, load_collision_meshes=True)
    robot = pk.Robot.from_urdf(urdf)
    collision = pk.collision.RobotCollision.from_urdf(urdf)
    hi, ri = get_humanoid_retarget_indices_g1_27dof()
    mask = create_conn_tree(robot, ri)
    plane = pk.collision.Heightmap(pose=jaxlie.SE3.identity(), size=jnp.array([.01, .01, 1.]),
                                   height_data=jnp.zeros((100, 100), dtype=jnp.float32))
    weights = dict(receipt['weights'])
    # Original v1 temporal weights rather than the prior 120 Hz sixfold variant.
    weights['root_smoothness'] = 2.0
    weights['foot_skating'] = 2.0
    names = list(robot.joints.actuated_names)
    starts = list(range(0, len(human) - 7, 4))
    if starts[-1] != len(human) - 8:
        starts.append(len(human) - 8)
    rows, previous = [], None
    began = time.monotonic()
    log = Path(__file__).with_name('windows.jsonl')
    with log.open('x', encoding='utf8', buffering=1) as stream:
        for wi, start in enumerate(starts):
            w = human[start:start + 8]
            indices = np.arange(start + 1, start + 8)
            left, right = foot_detect(w[None], thres=(.05 / fps) ** 2)
            target = jnp.asarray(w[1:])
            footl = plane.project_points(target[:, 10, :])
            footr = plane.project_points(target[:, 11, :])
            root, joints = solve_retargeting(robot=robot, robot_coll=collision, target_keypoints=target,
                is_left_foot_contact=jnp.asarray(left[0] > 0), is_right_foot_contact=jnp.asarray(right[0] > 0),
                left_foot_keypoints=footl, right_foot_keypoints=footr, smpl_joint_retarget_indices=hi,
                g1_joint_retarget_indices=ri, smpl_mask=mask, heightmap=plane, weights=weights,
                pick_point=jnp.zeros(3))
            q, rq = np.asarray(joints), np.asarray(root.wxyz_xyz)
            if q.shape != (7, 27) or rq.shape != (7, 7) or not np.isfinite(q).all() or not np.isfinite(rq).all():
                raise ValueError('invalid output')
            if not np.allclose(np.linalg.norm(rq[:, :4], axis=1), 1., atol=1e-4, rtol=0):
                raise ValueError('nonunit root quaternion')
            overlap = None
            if previous is not None:
                common = np.intersect1d(previous['indices'], indices)
                old = np.searchsorted(previous['indices'], common)
                new = np.searchsorted(indices, common)
                a = Rotation.from_quat(previous['root'][old][:, [1, 2, 3, 0]])
                b = Rotation.from_quat(rq[new][:, [1, 2, 3, 0]])
                overlap = {'sample_indices': common.tolist(),
                    'root_m': float(np.linalg.norm(previous['root'][old, 4:] - rq[new, 4:], axis=1).max()),
                    'rotation_rad': float((a.inv() * b).magnitude().max()),
                    'joint_rad': float(np.abs(previous['q'][old] - q[new]).max())}
            output = Path(__file__).with_name(f'window_{wi:02d}.npz')
            with output.open('xb') as f:
                np.savez_compressed(f, sample_indices=indices, source_frames=source_frames[indices],
                    source_times=times[indices], root_wxyz_xyz=rq, joint_pos=q, joint_names=np.asarray(names))
            row = {'window': wi, 'artifact': output.name, 'artifact_sha256': sha(output),
                   'sample_indices': indices.tolist(), 'overlap': overlap}
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print('WINDOW', wi + 1, 'of', len(starts), flush=True)
            previous = {'indices': indices, 'root': rq, 'q': q}
            rows.append(row)
    if sha(source) != expected:
        raise RuntimeError('source changed during run')
    report = {'source_sha256': expected, 'helper_sha256': sha(helper), 'urdf_sha256': sha(urdf_path),
              'script_sha256': sha(__file__), 'fps_reference_derived': fps, 'weights': weights,
              'source_indices_covered': [rows[0]['sample_indices'][0], rows[-1]['sample_indices'][-1]],
              'window_count': len(rows), 'rows_sha256': sha(log), 'elapsed_s': time.monotonic() - began,
              'training_ready': False,
              'limitations': ['one short SSM candidate, marker evidence inherited', 'SMPL-X not proven SMPL+H',
                  'old v1 helper and fixed objective; assumed ground and low speed proxy',
                  'overlapping windows not stitched', 'no verified caption, split, license or physical acceptance']}
    with Path(__file__).with_name('report.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
