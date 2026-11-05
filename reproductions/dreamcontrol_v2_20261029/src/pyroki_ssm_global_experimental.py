"""CPU single-trajectory PyRoki solve for the audited SSM 000008 witness.

Full sequence eliminates window stitching but uses the old v1 helper and an
assumed floor. Its output is experimental, not an accepted training sample.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
VARIANT = ROOT / 'pyroki_grab_temporal_20261030'
sys.path.insert(0, str(VARIANT))
sys.path.insert(1, str(ROOT / 'foot_contact_fit_20261030'))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof, create_conn_tree, solve_retargeting, foot_detect
from g1_fk import URDFFK


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
        raise ValueError('source, helper or URDF changed')
    with np.load(source, allow_pickle=False) as d:
        p = d['original_joints_humanml'].astype(np.float32)
        source_frames = d['source_frames'].copy()
        times = d['source_times'].copy()
    if p.shape != (43, 22, 3) or not np.isfinite(p).all() or not np.all(np.diff(source_frames) == 6):
        raise ValueError('unexpected source')
    human = np.stack((-p[..., 0], p[..., 2], p[..., 1]), axis=-1)
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
    weights['root_smoothness'] = weights['foot_skating'] = 2.0
    left, right = foot_detect(human[None], thres=(.05 / fps) ** 2)
    target = jnp.asarray(human[1:])
    started = time.monotonic()
    root, joints = solve_retargeting(robot=robot, robot_coll=collision, target_keypoints=target,
        is_left_foot_contact=jnp.asarray(left[0] > 0), is_right_foot_contact=jnp.asarray(right[0] > 0),
        left_foot_keypoints=plane.project_points(target[:, 10, :]),
        right_foot_keypoints=plane.project_points(target[:, 11, :]),
        smpl_joint_retarget_indices=hi, g1_joint_retarget_indices=ri, smpl_mask=mask,
        heightmap=plane, weights=weights, pick_point=jnp.zeros(3))
    q, rq = np.asarray(joints), np.asarray(root.wxyz_xyz)
    if q.shape != (42, 27) or rq.shape != (42, 7) or not np.isfinite(q).all() or not np.isfinite(rq).all():
        raise ValueError('invalid solver output')
    if not np.allclose(np.linalg.norm(rq[:, :4], axis=1), 1., atol=1e-4, rtol=0):
        raise ValueError('nonunit root quaternion')
    names = list(robot.joints.actuated_names)
    fk = URDFFK(urdf_path)
    links = ['left_hip_pitch_link', 'right_hip_pitch_link', 'left_knee_link', 'right_knee_link',
             'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_shoulder_pitch_link',
             'right_shoulder_pitch_link', 'left_elbow_link', 'right_elbow_link',
             'left_wrist_yaw_link', 'right_wrist_yaw_link']
    actual = fk.forward(rq[:, 4:], rq[:, [1, 2, 3, 0]], q, names, links=links)
    errors = np.linalg.norm(actual - human[1:, [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]], axis=-1)
    out = Path(__file__).with_name('trajectory.npz')
    with out.open('xb') as f:
        np.savez_compressed(f, sample_indices=np.arange(1, 43), source_frames=source_frames[1:],
            source_times=times[1:], root_wxyz_xyz=rq, joint_pos=q, joint_names=np.asarray(names))
    if sha(source) != expected:
        raise RuntimeError('source changed during solve')
    report = {'source_sha256': expected, 'helper_sha256': sha(helper), 'urdf_sha256': sha(urdf_path),
              'script_sha256': sha(__file__), 'artifact_sha256': sha(out), 'fps_reference_derived': fps,
              'weights': weights, 'samples': len(q), 'elapsed_s': time.monotonic() - started,
              'unscaled_world_point_mean_m': float(errors.mean()),
              'unscaled_world_point_max_m': float(errors.max()), 'training_ready': False,
              'limitations': ['one candidate, marker evidence inherited', 'SMPL-X not proven SMPL+H',
                  'old v1 helper with linear foot residual, assumed ground and low speed proxy',
                  'world error ignores human/robot morphology',
                  'no verified caption, split, license or physical acceptance']}
    with Path(__file__).with_name('report.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
