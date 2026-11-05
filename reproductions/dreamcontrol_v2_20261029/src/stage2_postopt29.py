"""Generated 22-point motions -> 29DoF G1 JAX/PyRoki fit and scene filters.

Robot-space position fitting has no human scale variable and never locks the
waist. Scene calibration and all filter thresholds are explicit caller input.
This is a documented implementation of the paper's outline, not author-v2 code.
"""
import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from scipy.spatial.transform import Rotation
import yourdfpy

R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0, str(R / 'training_disk_guard_20261101'))
from g1_fk import PAPER_22_LINKS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--sample-sha256', required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or sha(args.sample) != args.sample_sha256:
        raise ValueError('new output directory and hash-pinned generated sample required')
    scene = json.loads(args.scene.read_text())
    u = Path(scene['urdf'])
    if sha(u) != '572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95':
        raise ValueError('29DoF URDF changed')
    with np.load(args.sample, allow_pickle=False) as f:
        points = np.asarray(f['canonical_points'][0], dtype=np.float32)
        features = np.asarray(f['features'][0])
    if points.ndim != 3 or points.shape[1:] != (22, 3) or not np.isfinite(points).all():
        raise ValueError('generated canonical points required')
    mat = np.asarray(scene['canonical_to_world_rotation'], dtype=float)
    translation = np.asarray(scene['canonical_to_world_translation'], dtype=float)
    if mat.shape != (3, 3) or not np.allclose(mat.T @ mat, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(mat), 1):
        raise ValueError('explicit proper canonical-to-world calibration required')
    target = points @ mat.T + translation
    n, fps = len(points), 20.
    urdf = yourdfpy.URDF.load(str(u), load_meshes=False, load_collision_meshes=True)
    robot = pk.Robot.from_urdf(urdf)
    collision = pk.collision.RobotCollision.from_urdf(urdf)
    names = tuple(robot.joints.actuated_names)
    if len(names) != 29 or 'waist_roll_joint' not in names or 'waist_pitch_joint' not in names:
        raise ValueError('full 29DoF G1 required')
    links = jnp.asarray([robot.links.names.index(name) for name in PAPER_22_LINKS])
    limits = {j.attrib['name']: (float(j.find('limit').attrib['lower']), float(j.find('limit').attrib['upper']))
              for j in ET.parse(u).getroot().findall('joint') if j.attrib['type'] in ('revolute', 'prismatic')}
    lower = np.asarray([limits[name][0] for name in names])
    upper = np.asarray([limits[name][1] for name in names])
    joints = robot.joint_var_cls(jnp.arange(n))
    roots = jaxls.SE3Var(jnp.arange(n))
    initial_q = jnp.broadcast_to(joints.default_factory(), (n, 29))
    left = target[:, 1] - target[:, 2]
    left[:, 2] = 0
    left /= np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-6)
    up = np.broadcast_to([0., 0., 1.], left.shape)
    forward = np.cross(left, up)
    basis = np.stack([forward, left, up], axis=-1)
    quaternions = Rotation.from_matrix(basis).as_quat()[:, [3, 0, 1, 2]]
    root_init = jnp.asarray(np.column_stack([quaternions, target[:, 0]]), dtype=jnp.float32)
    plane = pk.collision.HalfSpace.from_point_and_normal([0., 0., scene['ground_z']], [0., 0., 1.])
    point_weights = jnp.ones((22, 1)).at[20:22].set(3.).at[jnp.asarray([7, 8, 10, 11])].set(2.)
    contacts = np.stack([features[:, -4:-2].mean(-1) > .5, features[:, -2:].mean(-1) > .5], axis=-1)
    feet = jnp.asarray([robot.links.names.index('left_ankle_roll_link'), robot.links.names.index('right_ankle_roll_link')])

    @jaxls.Cost.create_factory
    def position_cost(vals, q, root, goal):
        pose = vals[root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
        return ((pose.translation()[links] - goal) * point_weights * 20.).flatten()

    @jaxls.Cost.create_factory
    def root_smooth(vals, current, previous):
        return (vals[previous].inverse() @ vals[current]).log() * 1.

    @jaxls.Cost.create_factory
    def floor_cost(vals, q, root):
        geometry = collision.at_config(robot, vals[q]).transform(vals[root])
        distance = pk.collision.collide(geometry, plane)
        return jnp.maximum(.002 - distance, 0.).flatten() * 10.

    @jaxls.Cost.create_factory
    def stance_cost(vals, q, root, stance):
        pose = vals[root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))
        heights = pose.translation()[feet, 2]
        return (heights - scene['ground_z'] - scene['ankle_ground_offset_m']) * stance * 5.

    @jaxls.Cost.create_factory
    def slip_cost(vals, q, root, old_q, old_root, stance):
        current = (vals[root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))).translation()[feet, :2]
        previous = (vals[old_root] @ jaxlie.SE3(robot.forward_kinematics(cfg=vals[old_q]))).translation()[feet, :2]
        return ((current - previous) * stance[:, None] * fps).flatten() * .5

    costs = [position_cost(joints, roots, jnp.asarray(target, dtype=jnp.float32)),
             pk.costs.limit_cost(jax.tree.map(lambda x: x[None], robot), joints, 1000.),
             pk.costs.rest_cost(joints, joints.default_factory()[None], jnp.full((1, 29), .05)),
             pk.costs.smoothness_cost(robot.joint_var_cls(jnp.arange(1, n)), robot.joint_var_cls(jnp.arange(n - 1)), jnp.array([.5])),
             root_smooth(jaxls.SE3Var(jnp.arange(1, n)), jaxls.SE3Var(jnp.arange(n - 1))),
             pk.costs.self_collision_cost(jax.tree.map(lambda x: x[None], robot),
                                          jax.tree.map(lambda x: x[None], collision), joints, margin=.005, weight=1.),
             floor_cost(joints, roots), stance_cost(joints, roots, jnp.asarray(contacts, dtype=jnp.float32)),
             slip_cost(robot.joint_var_cls(jnp.arange(1, n)), jaxls.SE3Var(jnp.arange(1, n)),
                       robot.joint_var_cls(jnp.arange(n - 1)), jaxls.SE3Var(jnp.arange(n - 1)),
                       jnp.asarray(contacts[1:] & contacts[:-1], dtype=jnp.float32))]
    initial = jaxls.VarValues.make([joints.with_value(initial_q), roots.with_value(jaxlie.SE3(root_init))])
    began = time.monotonic()
    solved = jaxls.LeastSquaresProblem(costs, [joints, roots]).analyze().solve(initial_vals=initial)
    q = np.asarray(solved[joints])
    root = np.asarray(solved[roots].wxyz_xyz)
    if q.shape != (n, 29) or root.shape != (n, 7) or not np.isfinite(q).all() or not np.isfinite(root).all():
        raise ValueError('invalid 29DoF solver output')
    all_poses = jax.vmap(lambda qq, rr: jaxlie.SE3(rr) @ jaxlie.SE3(robot.forward_kinematics(cfg=qq)))(jnp.asarray(q), jnp.asarray(root))
    world = np.asarray(all_poses.translation())
    selected = world[:, np.asarray(links)]
    self_distances = np.asarray(jax.vmap(lambda qq: collision.compute_self_collision_distance(robot, qq))(jnp.asarray(q)))
    ground_distances = np.asarray(jax.vmap(lambda qq, rr: pk.collision.collide(collision.at_config(robot, qq).transform(jaxlie.SE3(rr)), plane))(jnp.asarray(q), jnp.asarray(root)))
    obstacle_distances = []
    for box in scene['obstacles']:
        geom = pk.collision.Box.from_extent(box['extent'], position=box['position'], wxyz=box.get('wxyz', [1., 0., 0., 0.]))
        distances = np.asarray(jax.vmap(lambda qq, rr: pk.collision.collide(collision.at_config(robot, qq).transform(jaxlie.SE3(rr)), geom))(jnp.asarray(q), jnp.asarray(root)))
        obstacle_distances.append(float(distances.min()))
    hand_errors, orientation_errors = [], []
    for goal in scene['hand_goals']:
        frame, index = goal['frame'], goal['point_index']
        hand_errors.append(float(np.linalg.norm(selected[frame, index] - np.asarray(goal['position']))))
        if 'wxyz' in goal:
            actual = np.asarray(all_poses.wxyz_xyz)[frame, int(links[index]), :4]
            angle = (Rotation.from_quat(actual[[1, 2, 3, 0]]).inv() * Rotation.from_quat(np.asarray(goal['wxyz'])[[1, 2, 3, 0]])).magnitude()
            orientation_errors.append(float(angle))
    foot_pos = world[:, np.asarray(feet)]
    foot_speed = np.linalg.norm(np.diff(foot_pos[..., :2], axis=0), axis=-1) * fps
    stance_intervals = contacts[1:] & contacts[:-1]
    slipping = int(((foot_speed > scene['max_stance_foot_speed_mps']) & stance_intervals).sum())
    stance_count = int(stance_intervals.sum())
    ratio = slipping / max(1, stance_count)
    floating_fraction = float(np.mean(np.min(foot_pos[..., 2], axis=1) > scene['ground_z'] + scene['max_floating_height_m']))
    violation = float(max(0., np.max(lower - q), np.max(q - upper)))
    checks = {'joint_limits': violation <= scene['joint_limit_tolerance_rad'],
              'root_height': bool(np.all((root[:, 6] >= scene['min_root_height_m']) & (root[:, 6] <= scene['max_root_height_m']))),
              'self_collision': float(self_distances.min()) >= -scene['collision_tolerance_m'],
              'ground_penetration': float(ground_distances.min()) >= -scene['collision_tolerance_m'],
              'foot_sliding': (stance_count > 0 or scene['allow_flight']) and ratio <= scene['max_slip_fraction'],
              'floating': scene['allow_flight'] or floating_fraction <= scene['max_floating_fraction'],
              'hand_target_error': not hand_errors or max(hand_errors) <= scene['max_hand_error_m'],
              'obstacle_collision': not obstacle_distances or min(obstacle_distances) >= -scene['collision_tolerance_m'],
              'wrist_orientation': not orientation_errors or max(orientation_errors) <= scene['max_wrist_angle_rad']}
    args.output_dir.mkdir(exist_ok=False)
    out = args.output_dir / 'robot29.npz'
    with out.open('xb') as f:
        np.savez_compressed(f, root_wxyz_xyz=root, joint_pos=q, joint_names=np.asarray(names),
                            target_world_points=target, fitted_world_points=selected,
                            input_frame_indices=np.arange(n), sample_times=np.arange(n) / fps,
                            foot_contact_prediction=contacts)
    report = {'generated_sample_sha256': sha(args.sample), 'scene_sha256': sha(args.scene), 'urdf_sha256': sha(u),
              'output_sha256': sha(out), 'joint_count': 29, 'waist_locked': False, 'frames': n, 'fps': fps,
              'solver_backend': str(jax.devices()[0]), 'optimization_elapsed_s': time.monotonic() - began,
              'position_fit_rmse_m': float(np.sqrt(np.mean((selected - target)**2))),
              'max_joint_limit_violation_rad': violation, 'minimum_self_collision_distance_m': float(self_distances.min()),
              'minimum_ground_collision_distance_m': float(ground_distances.min()), 'obstacle_min_distances_m': obstacle_distances,
              'stance_slipping_fraction': ratio, 'floating_fraction': floating_fraction, 'hand_target_errors_m': hand_errors,
              'predicted_stance_intervals': stance_count,
              'wrist_orientation_errors_rad': orientation_errors, 'checks': checks,
              'passed_requested_kinematic_filters': all(checks.values()),
              'eligible_for_calibrated_scene_reference': all(checks.values()) and scene['calibration_verified'],
              'not_applicable_filters': (['obstacle_collision'] if not scene['obstacles'] else []) + (['wrist_orientation'] if not orientation_errors else []),
              'rejection_reasons': [k for k, v in checks.items() if not v],
              'scene_calibration_verified': scene['calibration_verified'], 'physics_tracking_tested': False,
              'scope': 'generated-motion postoptimization/heuristic filters only; no GRAB/Omni source filtering',
              'limitations': ['Project-chosen objective weights and thresholds implement the published outline, not released v2 code.',
                              'Foot stance is predicted from generated contact features, not measured contact.',
                              'Collision checks use PyRoki URDF collision representation; this is not an RL tracking test.',
                              '263D keypoint fitting does not uniquely recover every one of the 29 joint angles.']}
    with (args.output_dir / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
