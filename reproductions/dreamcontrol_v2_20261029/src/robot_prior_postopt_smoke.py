"""CPU wiring smoke against fixed old PyRoki and an existing 8-frame witness.

This does not generate training data, validate physics, or implement author v2.
Run in a fresh directory with JAX_PLATFORMS=cpu and CUDA_VISIBLE_DEVICES empty.
"""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from robot_prior_postopt import postopt_canonical_motion

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PROJECT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
VARIANT = ROOT / 'pyroki_grab_temporal_20261030'
sys.path.insert(0, str(VARIANT))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof, create_conn_tree, solve_retargeting, foot_detect


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def main():
    if any(d.platform != 'cpu' for d in jax.devices()): raise RuntimeError('CPU required')
    source = ROOT / 'humanml_mirror_time_witness_v2_20261030/000008_joint_witness.npz'
    baseline = ROOT / 'pyroki_ssm_000008_human_20261031/window_00.npz'
    helper = VARIANT / 'retarget_helpers/_utils.py'
    urdf_path = PROJECT / 'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
    receipt = json.loads((VARIANT / 'report.json').read_text())
    if sha(source) != '8354f592da84aa9e62b87f4ab467465cd8a43120e39d5c8ae346966a8883a0b7' or sha(helper) != receipt['helper_sha256'] or sha(urdf_path) != receipt['urdf_sha256'] or sha(baseline) != '1dfb33432bc9a33f0a134bc563dfbad1c417e3602c4332cdb1032035aab0be62':
        raise ValueError('input hash mismatch')
    with np.load(source, allow_pickle=False) as f:
        points = f['original_joints_humanml'][:8].astype(np.float32)
    urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, load_collision_meshes=True)
    robot = pk.Robot.from_urdf(urdf)
    collision = pk.collision.RobotCollision.from_urdf(urdf)
    hi, ri = get_humanoid_retarget_indices_g1_27dof()
    mask = create_conn_tree(robot, ri)
    ground = pk.collision.Heightmap(pose=jaxlie.SE3.identity(), size=jnp.array([.01, .01, 1.]),
                                    height_data=jnp.zeros((100, 100), dtype=jnp.float32))
    names = list(robot.joints.actuated_names)
    limits = {j.attrib['name']: (float(j.find('limit').attrib['lower']), float(j.find('limit').attrib['upper']))
              for j in ET.parse(urdf_path).getroot().findall('joint')
              if j.find('limit') is not None and 'lower' in j.find('limit').attrib}
    lower, upper = zip(*(limits[n] for n in names))
    weights = dict(receipt['weights']); weights['root_smoothness'] = 2.; weights['foot_skating'] = 2.
    fps = 20.00000762939453  # Same reference-derived nominal rate as previous isolated window.
    result = postopt_canonical_motion(points, solver=solve_retargeting, foot_detector=foot_detect,
        robot=robot, robot_collision=collision, heightmap=ground, human_indices=hi, robot_indices=ri,
        connection_mask=mask, weights=weights, canonical_to_solver_rotation=[[-1,0,0],[0,0,1],[0,1,0]],
        canonical_to_solver_translation=[0,0,0], pick_point_solver=[0,0,0],
        foot_displacement_sq_threshold=(.05/fps)**2, joint_names=names, joint_lower=lower, joint_upper=upper, fps=fps)
    with np.load(baseline, allow_pickle=False) as old:
        if not np.array_equal(old['sample_indices'], result['input_frame_indices']) or tuple(old['joint_names']) != tuple(names):
            raise ValueError('baseline indices or joint order changed')
        q_diff = float(np.max(np.abs(old['joint_pos'] - result['joint_pos'])))
        root_diff = float(np.max(np.abs(old['root_wxyz_xyz'] - result['root_wxyz_xyz'])))
    if max(q_diff, root_diff) > 1e-4:
        raise ValueError(f'bridge differs from baseline: joint {q_diff}, root {root_diff}')
    report = {'source_sha256': sha(source), 'helper_sha256': sha(helper), 'urdf_sha256': sha(urdf_path),
              'bridge_sha256': sha(Path(__file__).with_name('robot_prior_postopt.py')),
              'script_sha256': sha(__file__), 'baseline_sha256': sha(baseline),
              'baseline_joint_max_abs_difference': q_diff, 'baseline_root_max_abs_difference': root_diff,
              'root_shape': list(result['root_wxyz_xyz'].shape),
              'joint_shape': list(result['joint_pos'].shape), 'input_frame_indices': result['input_frame_indices'].tolist(),
              'metadata': result['metadata'], 'devices': [str(d) for d in jax.devices()],
              'limitations': ['Existing human witness, not a generated model sample.',
                              'Old approximate solver, assumed ground and low-speed proxies.',
                              'No contact, dynamics, source licensing or training acceptance.']}
    with Path('report.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
