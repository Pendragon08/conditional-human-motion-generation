"""Motion-preserving two-stage 29DoF postoptimization for repaired prior v2.

Stage A realizes decoded points. Stage B starts there and projects toward task
constraints, with bounded deformation and unchanged physical admission gates.
Penalty residuals do not guarantee feasibility: independent CPU FK gates decide.
JAX/PyRoki imports are lazy; validation helpers run on CPU with NumPy only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

URDF_SHA256 = "572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95"
TASKS = ("pick", "pour", "bimanual_box_lift", "box_push_pull")
WRISTS = (20, 21)
FEET = (10, 11)


@dataclass(frozen=True)
class PostoptPolicyV2:
    # Existing strict supported-task admission gates, also used for repaired Pick.
    wrist_position_max_m: float = .03
    wrist_angle_max_rad: float = .20
    joint_projection_max_rad: float = .001
    joint_violation_max_rad: float = 1e-7
    ground_penetration_max_m: float = .005
    foot_sole_max_m: float = .02
    stance_speed_threshold_mps: float = .15
    stance_slipping_fraction_max: float = .1
    root_min_m: float = .3
    root_max_m: float = 1.2
    # New explicit project quality gates. These are not claimed paper parameters.
    decoded_points_rmse_max_m: float = .05
    decoded_points_peak_max_m: float = .15
    decoded_velocity_rmse_max_mps: float = .20
    stage_b_nonwrist_rmse_max_m: float = .03
    stage_b_nonwrist_peak_max_m: float = .08
    stage_b_root_shift_max_m: float = .05
    stage_b_joint_shift_max_rad: float = .50
    joint_frame_step_max_rad: float = .30
    wrist_frame_angle_max_rad: float = .25
    # Pre-filter prevents task templates replacing severely uncontrolled output.
    raw_wrist_target_max_m: float = .12
    solver_max_iterations: int = 200
    disk_reserve_bytes: int = 8 * 1024**3 + 456 * 1024**2


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canonical_json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rms_vector(delta):
    return float(np.sqrt(np.mean(np.sum(np.asarray(delta, dtype=float)**2, axis=-1))))


def validate_coordinate_metadata(metadata, canonical_targets, world_targets):
    """Check the sampler's explicit transform; never infer/reselect its floor."""
    r = np.asarray(metadata["canonical_to_world_rotation"], dtype=float)
    t = np.asarray(metadata["canonical_to_world_translation"], dtype=float)
    inv_r = np.asarray(metadata["world_to_canonical_rotation"], dtype=float)
    inv_t = np.asarray(metadata["world_to_canonical_translation"], dtype=float)
    if r.shape != (3, 3) or inv_r.shape != (3, 3) or t.shape != (3,) or inv_t.shape != (3,):
        raise ValueError("Explicit forward and inverse 3D transforms are required")
    if not all(np.isfinite(x).all() for x in (r, t, inv_r, inv_t)):
        raise ValueError("Nonfinite coordinate transform")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1., atol=1e-7):
        raise ValueError("Canonical rotation is not a proper rotation")
    if not np.allclose(inv_r, r.T, atol=1e-7) or not np.allclose(inv_t, -r.T @ t, atol=1e-7):
        raise ValueError("Forward/inverse coordinate transforms disagree")
    world = np.asarray(world_targets, dtype=np.float32)
    if world.shape != np.asarray(canonical_targets).shape or not np.isfinite(world).all():
        raise ValueError("Saved world target shape/values invalid")
    calculated = np.asarray(canonical_targets, dtype=float) @ r.T + t
    error = float(np.max(np.abs(calculated - world)))
    if error > 1e-6:
        raise ValueError("Canonical/world target round-trip differs by more than 1 um")
    expected = hashlib.sha256(np.ascontiguousarray(world).tobytes()).hexdigest()
    if metadata["target_world_content_sha256"] != expected:
        raise ValueError("Saved original world-target SHA mismatch")
    provenance = metadata["coordinate_provenance"]
    if provenance.get("schema") != "training_floor_canonical_v2":
        raise ValueError("Repaired coordinate provenance is required")
    for name in ("ground_z", "height_origin_world_m"):
        if not np.isfinite(float(provenance[name])):
            raise ValueError("Nonfinite coordinate provenance: " + name)
    for name in ("calibration_sha256", "joint_order_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", str(provenance[name])):
            raise ValueError("Missing SHA provenance: " + name)
    return {"rotation": r, "translation": t, "ground_z": float(provenance["ground_z"]),
            "target_roundtrip_max_abs_m": error, "coordinate_provenance": provenance}


def load_sample(sample, expected_sha):
    if sha_file(sample) != expected_sha:
        raise ValueError("Generated source sample SHA mismatch")
    with np.load(sample, allow_pickle=False) as z:
        m = json.loads(z["metadata_utf8"].tobytes())
        points = np.asarray(z["canonical_points"], dtype=np.float64)
        targets = np.asarray(z["target_points"], dtype=np.float64)
        world_targets = np.asarray(z["target_points_world"], dtype=np.float32)
        mask = np.asarray(z["target_mask"])
        orientation = np.asarray(z["desired_wrist_quat_wxyz"], dtype=np.float64)
        closed = np.asarray(z["planned_hand_closed"])
        objects = np.asarray(z["desired_object_positions"], dtype=np.float64)
    task = m["task"]
    if task not in TASKS or float(m["fps"]) != 20.:
        raise ValueError("Only authorized four tasks on the 20 Hz grid are supported")
    n = 120 if task == "pick" else 180
    if points.shape != (1, n, 22, 3) or targets.shape != (n, 22, 3) or mask.shape != (n, 22):
        raise ValueError("Sample point/mask dimensions disagree")
    if orientation.shape != (n, 2, 4) or closed.shape != (n, 2) or objects.shape != (n, 3):
        raise ValueError("Explicit orientation, hand and object plans are required")
    if not all(np.isfinite(a).all() for a in (points, targets, orientation, objects, closed)):
        raise ValueError("Nonfinite generated sample")
    if not np.isin(mask, [0, 1]).all() or not np.isin(closed, [0, 1]).all():
        raise ValueError("Masks and hand plans must be binary")
    if not np.allclose(np.linalg.norm(orientation, axis=-1), 1., atol=1e-5):
        raise ValueError("Desired wrist quaternions must be normalized wxyz")
    calibration = validate_coordinate_metadata(m, targets, world_targets)
    decoded_world = points[0] @ calibration["rotation"].T + calibration["translation"]
    active = mask[:, WRISTS].astype(bool)
    expected_hands = (False, True) if task in ("pick", "pour") else (True, True)
    if tuple(active.any(axis=0).tolist()) != expected_hands:
        raise ValueError("Required task wrist masks are missing or swapped")
    frame = int(m["grasp_frame"])
    if not 0 <= frame < n or abs(float(m["grasp_time_s"]) - frame / 20.) > 1e-6:
        raise ValueError("Grasp frame/time metadata disagree")
    return dict(metadata=m, decoded_world=decoded_world, targets_world=world_targets,
                wrist_mask=active, orientations=orientation, hand_closed=closed.astype(np.int8),
                desired_objects=objects, frames=n, fps=20., calibration=calibration)


def deformation_metrics(decoded, final_points, stage_a_points, q, qa, root, root_a, fps=20.):
    decoded = np.asarray(decoded, dtype=float)
    points = np.asarray(final_points, dtype=float)
    other = np.arange(20)  # Keep both task wrists out of the body-change statistic.
    original_delta = points - decoded
    projection_delta = points[:, other] - np.asarray(stage_a_points)[:, other]
    return {
        "decoded_point_vector_rmse_m": rms_vector(original_delta),
        "decoded_point_max_l2_m": float(np.linalg.norm(original_delta, axis=-1).max()),
        "decoded_velocity_vector_rmse_mps": rms_vector(np.diff(original_delta, axis=0) * fps),
        "stage_b_nonwrist_vector_rmse_m": rms_vector(projection_delta),
        "stage_b_nonwrist_max_l2_m": float(np.linalg.norm(projection_delta, axis=-1).max()),
        "stage_b_root_max_shift_m": float(np.linalg.norm(np.asarray(root)[:, 4:] - np.asarray(root_a)[:, 4:], axis=-1).max()),
        "stage_b_joint_max_shift_rad": float(np.max(np.abs(np.asarray(q) - qa))),
        "joint_max_frame_step_rad": float(np.max(np.abs(np.diff(q, axis=0)))),
    }


def acceptance_checks(metrics, policy=PostoptPolicyV2()):
    """Strict gates, independent of optimizer residual weights and task labels."""
    p = policy
    if not all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
        return {"finite": False}
    return {
        "finite": bool(metrics["finite"]),
        "joint_projection": metrics["joint_projection_rad"] < p.joint_projection_max_rad,
        "joint_limits": metrics["joint_violation_rad"] < p.joint_violation_max_rad,
        "hand_target": metrics["wrist_max_l2_m"] < p.wrist_position_max_m,
        "wrist_orientation": metrics["wrist_max_angle_rad"] < p.wrist_angle_max_rad,
        "ground_mesh_plane": metrics["minimum_mesh_height_above_ground_m"] >= -p.ground_penetration_max_m,
        "feet_on_ground": metrics["maximum_foot_sole_height_above_ground_m"] < p.foot_sole_max_m,
        "stance_foot_speed": metrics["stance_slipping_fraction"] <= p.stance_slipping_fraction_max,
        "root_height": metrics["root_min_height_above_ground_m"] > p.root_min_m and metrics["root_max_height_above_ground_m"] < p.root_max_m,
        "table_vertex_penetration": metrics["minimum_table_vertex_sdf_m"] >= -p.ground_penetration_max_m,
        "preserve_decoded_points": metrics["decoded_point_vector_rmse_m"] <= p.decoded_points_rmse_max_m and metrics["decoded_point_max_l2_m"] <= p.decoded_points_peak_max_m,
        "preserve_decoded_velocity": metrics["decoded_velocity_vector_rmse_mps"] <= p.decoded_velocity_rmse_max_mps,
        "bounded_nonwrist_correction": metrics["stage_b_nonwrist_vector_rmse_m"] <= p.stage_b_nonwrist_rmse_max_m and metrics["stage_b_nonwrist_max_l2_m"] <= p.stage_b_nonwrist_peak_max_m,
        "bounded_root_correction": metrics["stage_b_root_max_shift_m"] <= p.stage_b_root_shift_max_m,
        "bounded_joint_correction": metrics["stage_b_joint_max_shift_rad"] <= p.stage_b_joint_shift_max_rad,
        "joint_temporal_continuity": metrics["joint_max_frame_step_rad"] <= p.joint_frame_step_max_rad,
        "wrist_temporal_continuity": metrics["wrist_max_frame_angle_rad"] <= p.wrist_frame_angle_max_rad,
    }


class MotionPreservingSolverV2:
    """Construct once per frame count; existing JAX/PyRoki environment is reused."""
    def __init__(self, urdf, frames, fk_module, policy=PostoptPolicyV2()):
        import jax
        import jax.numpy as jnp
        import jaxlie
        import jaxls
        import pyroki as pk
        import yourdfpy
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation
        if sha_file(urdf) != URDF_SHA256:
            raise ValueError("Pinned 29DoF URDF identity changed")
        spec = importlib.util.spec_from_file_location("repair_postopt_readonly_fk", fk_module)
        fk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fk)
        self.fk, self.urdf, self.fk_module = fk.URDFFK(urdf), Path(urdf), Path(fk_module)
        self.policy, self.frames = policy, frames
        self.Rotation, self.jax, self.jnp, self.jaxlie = Rotation, jax, jnp, jaxlie
        u = yourdfpy.URDF.load(str(urdf), load_meshes=False, load_collision_meshes=True)
        robot = pk.Robot.from_urdf(u)
        names = list(robot.joints.actuated_names)
        if len(names) != 29 or "waist_roll_joint" not in names or "waist_pitch_joint" not in names:
            raise ValueError("Full 29DoF G1 joint ordering required")
        self.robot, self.names = robot, names
        links = jnp.array([robot.links.names.index(k) for k in fk.PAPER_22_LINKS])
        wrists = jnp.array([robot.links.names.index(side + "_wrist_yaw_link") for side in ("left", "right")])
        feet = jnp.array([robot.links.names.index(side + "_ankle_roll_link") for side in ("left", "right")])
        q0 = np.zeros(29, np.float32)
        for side in ("left", "right"):
            for part, value in [("hip_pitch", -.2), ("knee", .42), ("ankle_pitch", -.23), ("elbow", .87), ("shoulder_pitch", .35)]:
                q0[names.index(side + "_" + part + "_joint")] = value
        self.q0 = q0  # Initialization only; never a root/stance template target.
        vertices, vertex_links, foot_vertex_indices = [], [], [[], []]
        self.meshes = {}
        for k, name in enumerate(robot.links.names):
            mesh = pk.collision.RobotCollision._get_trimesh_collision_geometries(u, name)
            full = np.asarray(mesh.vertices)
            if not len(full):
                continue
            self.meshes[name] = full.copy()
            try:
                hull = full[ConvexHull(full).vertices]
            except Exception:
                hull = full
            begin = len(vertices)
            vertices.extend(hull)
            vertex_links.extend([k] * len(hull))
            for side_index, side in enumerate(("left", "right")):
                if name == side + "_ankle_roll_link":
                    foot_vertex_indices[side_index].extend(range(begin, begin + len(hull)))
        if not all(foot_vertex_indices):
            raise ValueError("Both ankle collision meshes are required")
        vtx, vlinks = jnp.asarray(np.asarray(vertices), dtype=jnp.float32), jnp.asarray(vertex_links)
        foot_vertices = [jnp.asarray(x) for x in foot_vertex_indices]
        self.all_links = sorted(self.fk.links)
        self.kp = [self.all_links.index(k) for k in fk.PAPER_22_LINKS]
        self.cpu_feet = [self.all_links.index(side + "_ankle_roll_link") for side in ("left", "right")]
        self.cpu_wrists = [self.all_links.index(side + "_wrist_yaw_link") for side in ("left", "right")]
        limits = {j.get("name"): j.find("limit") for j in ET.parse(urdf).getroot().findall("joint") if j.get("type") == "revolute"}
        self.low = np.array([float(limits[name].get("lower")) for name in names])
        self.high = np.array([float(limits[name].get("upper")) for name in names])
        qvars, rvars = robot.joint_var_cls(jnp.arange(frames)), jaxls.SE3Var(jnp.arange(frames))

        def pose(vals, q, r):
            return vals[r] @ jaxlie.SE3(robot.forward_kinematics(cfg=vals[q]))

        @jaxls.Cost.create_factory
        def decoded_cost(vals, q, r, target):
            return ((pose(vals, q, r).translation()[links] - target) / .03).flatten()

        @jaxls.Cost.create_factory
        def decoded_velocity_cost(vals, q, r, qp, rp, velocity):
            value = (pose(vals, q, r).translation()[links] - pose(vals, qp, rp).translation()[links]) * 20.
            return ((value - velocity) / .20).flatten()

        @jaxls.Cost.create_factory
        def task_cost(vals, q, r, goal, orientation, mask, penalty):
            body = pose(vals, q, r)
            wrist_pose = jaxlie.SE3(body.wxyz_xyz[wrists])
            position = (wrist_pose.translation() - goal) / .03
            angle = (jaxlie.SO3(orientation).inverse() @ wrist_pose.rotation()).log() / .20
            return jnp.concatenate([(position * mask[:, None]).flatten(), (angle * mask[:, None]).flatten()]) * penalty

        @jaxls.Cost.create_factory
        def stage_a_preservation(vals, q, r, pa, qa, ra):
            current = pose(vals, q, r).translation()[links]
            # Uniform body anchoring to decoded realization, not a task template.
            fit = (current - pa) / .03
            root_shift = (vals[r].translation() - ra[4:]) / .05
            q_shift = (vals[q] - qa) / .5
            return jnp.concatenate([fit.flatten(), root_shift, q_shift])

        @jaxls.Cost.create_factory
        def trust_region_cost(vals, q, r, pa, qa, ra):
            delta = pose(vals, q, r).translation()[links[:20]] - pa[:20]
            keypoint = jnp.maximum(jnp.sqrt(jnp.maximum(jnp.sum(delta ** 2, axis=-1), 1e-12)) - .075, 0.) / .005
            root = jnp.maximum(jnp.sqrt(jnp.maximum(jnp.sum((vals[r].translation() - ra[4:]) ** 2), 1e-12)) - .045, 0.) / .005
            joint = jnp.maximum(jnp.abs(vals[q] - qa) - .45, 0.) / .05
            return jnp.concatenate([keypoint, root[None], joint]) * 4.

        @jaxls.Cost.create_factory
        def geometry_cost(vals, q, r, ground_z, penalty, table_scale):
            body = pose(vals, q, r)
            world = jaxlie.SE3(body.wxyz_xyz[vlinks]).apply(vtx)
            ground = jnp.maximum(ground_z + .001 - jnp.min(world[:, 2]), 0.) / .005
            soles = jnp.stack([jnp.min(world[ii, 2]) - ground_z for ii in foot_vertices])
            foot = jnp.maximum(soles - .015, 0.) / .005
            # Existing tabletop dimensions; world height is independent of canonical floor.
            d = jnp.abs(world - jnp.array([.45, 0., .4])) - jnp.array([.3, .3, .4])
            sdf = jnp.linalg.norm(jnp.maximum(d, 0.) + 1e-9, axis=-1) + jnp.minimum(jnp.max(d, axis=-1), 0.)
            table = jnp.maximum(.002 - jnp.min(sdf), 0.) / .005
            return jnp.concatenate([ground[None], foot, table[None] * table_scale]) * penalty

        @jaxls.Cost.create_factory
        def continuity_cost(vals, q, r, qp, rp):
            dq = jnp.maximum(jnp.abs(vals[q] - vals[qp]) - .25, 0.) / .05
            speed = jnp.linalg.norm(pose(vals, q, r).translation()[feet, :2] - pose(vals, qp, rp).translation()[feet, :2], axis=-1) * 20.
            slip = jnp.maximum(speed - .10, 0.) / .05
            current = jaxlie.SO3(pose(vals, q, r).rotation().wxyz[wrists])
            previous = jaxlie.SO3(pose(vals, qp, rp).rotation().wxyz[wrists])
            delta = (previous.inverse() @ current).log()
            speed = jnp.sqrt(jnp.maximum(jnp.sum(delta**2, axis=-1), 1e-12))
            wrist_step = jnp.maximum(speed - .20, 0.) / .05
            return jnp.concatenate([dq, slip, wrist_step])

        @jaxls.Cost.create_factory
        def wrist_gauge_cost(vals, q, r, orientation, mask):
            current = jaxlie.SO3(pose(vals, q, r).rotation().wxyz[wrists])
            angle = (jaxlie.SO3(orientation).inverse() @ current).log()
            return (angle * mask[:, None] * 2.5).flatten()

        def solve(target, goals, orientations, mask, iq, ir, pa, qa, ra, ground_z, stage, penalty, table_scale=1.):
            q, r = qvars, rvars
            qc, qp = robot.joint_var_cls(jnp.arange(1, frames)), robot.joint_var_cls(jnp.arange(frames - 1))
            rc, rp = jaxls.SE3Var(jnp.arange(1, frames)), jaxls.SE3Var(jnp.arange(frames - 1))
            costs = [decoded_cost(q, r, target),
                     decoded_velocity_cost(qc, rc, qp, rp, jnp.diff(target, axis=0) * 20.),
                     geometry_cost(q, r, jnp.full((frames,), ground_z), jnp.full((frames,), 4.), jnp.full((frames,), table_scale)),
                     continuity_cost(qc, rc, qp, rp),
                     pk.costs.limit_cost(jax.tree.map(lambda x: x[None], robot), q, 10000.)]
            if stage == "A":
                costs.append(wrist_gauge_cost(q, r, orientations, mask))
                # Only a tiny joint gauge prior for point-unobservable twists.
                costs.append(pk.costs.rest_cost(q, jnp.asarray(q0)[None], jnp.full((1, 29), .01)))
            else:
                costs.extend([task_cost(q, r, goals, orientations, mask, jnp.full((frames,), penalty)),
                              stage_a_preservation(q, r, pa, qa, ra), trust_region_cost(q, r, pa, qa, ra)])
            initial = jaxls.VarValues.make([q.with_value(iq), r.with_value(jaxlie.SE3(ir))])
            solved = jaxls.LeastSquaresProblem(costs, [q, r]).analyze().solve(
                initial_vals=initial, linear_solver=jaxls.ConjugateGradientConfig(tolerance_max=1e-5),
                trust_region=jaxls.TrustRegionConfig(lambda_max=1e12),
                termination=jaxls.TerminationConfig(max_iterations=policy.solver_max_iterations,
                                                    cost_tolerance=1e-9, parameter_tolerance=1e-9))
            return solved[q], solved[r].wxyz_xyz
        self.solve = jax.jit(solve, static_argnames=("stage",))

    def initial_state(self, decoded):
        # Root position and facing follow each decoded frame, no fixed-root template.
        lateral = decoded[:, 1] - decoded[:, 2]
        lateral[:, 2] = 0.
        norms = np.linalg.norm(lateral, axis=-1, keepdims=True)
        if np.min(norms) < 1e-5:
            raise ValueError("Decoded hip axis is degenerate")
        lateral /= norms
        up = np.broadcast_to([0., 0., 1.], lateral.shape)
        forward = np.cross(lateral, up)
        rotation = np.stack([forward, lateral, up], axis=-1)
        quat = self.Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
        return np.broadcast_to(self.q0, (self.frames, 29)).copy(), np.c_[quat, decoded[:, 0]].astype(np.float32)

    def cpu_state(self, raw_q, raw_root):
        if not np.isfinite(raw_q).all() or not np.isfinite(raw_root).all():
            raise ValueError("Nonfinite solver output preserved")
        q = np.clip(np.asarray(raw_q, dtype=float), self.low + 1e-6, self.high - 1e-6).astype(np.float32)
        projection = float(np.max(np.abs(q - raw_q)))
        points, rotations = self.fk.forward(raw_root[:, 4:], raw_root[:, [1, 2, 3, 0]], q,
                                           self.names, links=self.all_links, return_rotations=True)
        return q, points, rotations, projection

    def measure(self, sample, raw_q, raw_root, stage_a):
        q, full, rot, projection = self.cpu_state(raw_q, raw_root)
        points = full[:, self.kp]
        mask = sample["wrist_mask"]
        error = np.linalg.norm(points[:, WRISTS] - sample["targets_world"][:, WRISTS], axis=-1)
        wquat = self.Rotation.from_matrix(rot[:, self.cpu_wrists].reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(self.frames, 2, 4)
        angle = 2 * np.arccos(np.clip(np.abs(np.sum(wquat * sample["orientations"], axis=-1)), 0., 1.))
        angular_step = 2 * np.arccos(np.clip(np.abs(np.sum(wquat[1:] * wquat[:-1], axis=-1)), 0., 1.))
        minima, foot_soles, table_distances = [], [], []
        ground = sample["calibration"]["ground_z"]
        for name, vertices in self.meshes.items():
            k = self.all_links.index(name)
            world = np.einsum("tij,vj->tvi", rot[:, k], vertices) + full[:, k, None]
            z = world[:, :, 2].min(axis=1) - ground
            minima.append(z)
            if k in self.cpu_feet:
                foot_soles.append(z)
            d = np.abs(world - [.45, 0., .4]) - [.3, .3, .4]
            sdf = np.linalg.norm(np.maximum(d, 0.), axis=-1) + np.minimum(np.max(d, axis=-1), 0.)
            table_distances.append(float(sdf.min()))
        speed = np.linalg.norm(np.diff(full[:, self.cpu_feet, :2], axis=0), axis=-1) * 20.
        metrics = {"finite": True, "joint_projection_rad": projection,
                   "joint_violation_rad": float(max(0., np.max(self.low - q), np.max(q - self.high))),
                   "wrist_max_l2_m": float(error[mask].max()), "wrist_mean_l2_m": float(error[mask].mean()),
                   "wrist_max_angle_rad": float(angle[mask].max()),
                   "wrist_max_frame_angle_rad": float(angular_step.max()),
                   "minimum_mesh_height_above_ground_m": float(np.stack(minima, axis=1).min()),
                   "maximum_foot_sole_height_above_ground_m": float(np.stack(foot_soles, axis=1).max()),
                   "stance_slipping_fraction": float(np.mean(speed > self.policy.stance_speed_threshold_mps)),
                   "max_stance_speed_mps": float(speed.max()),
                   "root_min_height_above_ground_m": float((raw_root[:, 6] - ground).min()),
                   "root_max_height_above_ground_m": float((raw_root[:, 6] - ground).max()),
                   "minimum_table_vertex_sdf_m": min(table_distances)}
        metrics.update(deformation_metrics(sample["decoded_world"], points, stage_a["points"], q,
                                           stage_a["q"], raw_root, stage_a["root"]))
        return metrics, q, points, wquat

    def process(self, sample_path, output, expected_sha):
        import shutil
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        report = {"schema": "motion_preserving_postopt_v2", "source_sample": str(sample_path),
                  "source_sample_sha256": expected_sha, "policy": asdict(self.policy),
                  "policy_sha256": canonical_json_sha(asdict(self.policy)), "stages": [],
                  "source_sha256": sha_file(__file__), "urdf_sha256": sha_file(self.urdf),
                  "fk_source_sha256": sha_file(self.fk_module),
                  "native_scene_collision_replay": "not_run", "eligible_for_calibrated_scene_reference": False,
                  "passed_requested_kinematic_checks": False, "checks": {},
                  "physical_grasp_verified": False, "object_task_success_measured": False,
                  "orientation_targets_enter_diffusion": False, "hand_plan_is_contact_evidence": False}
        def finish(status, reason=None):
            report.update(status=status, elapsed_seconds=time.monotonic() - start)
            if reason:
                report["rejection_reason"] = reason
            with (output / "report.json").open("x", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            return report
        try:
            if shutil.disk_usage(output).free <= self.policy.disk_reserve_bytes:
                raise ValueError("Insufficient postoptimization disk reserve")
            sample = load_sample(sample_path, expected_sha)
            m = sample["metadata"]
            joint_sha = hashlib.sha256(json.dumps(self.names, separators=(",", ":")).encode()).hexdigest()
            if m["coordinate_provenance"]["joint_order_sha256"] != joint_sha:
                raise ValueError("Sampler/solver joint-order binding differs")
            if sample["frames"] != self.frames:
                raise ValueError("Solver frame count differs from sample")
            report.update(task=m["task"], candidate_id=m["candidate_id"], frames=self.frames, joint_count=29,
                          source_initial_noise_sha256=m["initial_noise_sha256"], source_feature_content_sha256=m["feature_content_sha256"],
                          scene_sha256=m["scene_sha256"], task_conditions_source_sha256=m["task_conditions_source_sha256"],
                          checkpoint_sha256=m["checkpoint_sha256"], task_object_position=m["task_object_position"],
                          coordinate_provenance=m["coordinate_provenance"],
                          target_roundtrip_max_abs_m=sample["calibration"]["target_roundtrip_max_abs_m"])
            raw_error = np.linalg.norm(sample["decoded_world"][:, WRISTS] - sample["targets_world"][:, WRISTS], axis=-1)[sample["wrist_mask"]]
            report["raw_wrist_target_max_m"] = float(raw_error.max())
            if raw_error.max() > self.policy.raw_wrist_target_max_m:
                return finish("generated_candidate_rejected", "raw_spatial_guidance_requires_excessive_task_rewrite")
            initial_q, initial_root = self.initial_state(sample["decoded_world"])
            jnp = self.jnp
            target = jnp.asarray(sample["decoded_world"], dtype=jnp.float32)
            goals = jnp.asarray(sample["targets_world"][:, WRISTS], dtype=jnp.float32)
            orient = jnp.asarray(sample["orientations"], dtype=jnp.float32)
            mask = jnp.asarray(sample["wrist_mask"], dtype=jnp.float32)
            ground = sample["calibration"]["ground_z"]
            # The old, immutable state supplies only an optimizer start.
            from scipy.spatial.transform import Slerp
            legacy = Path("/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103/reference_scale500_20261104/final_library_v1/reference_bank.json")
            assert sha_file(legacy) == "01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572"
            rows = json.loads(legacy.read_text())["references"]
            assert self.frames == 120
            clock = np.arange(self.frames)
            contact = int(m["grasp_frame"])
            assert 40 <= contact <= 50
            phase = np.interp(clock, [0,contact,contact+40,119], [0,40,80,119])
            lo = np.floor(phase).astype(int)
            hi = np.minimum(lo+1,119)
            blend = (phase-lo)[:,None,None]
            center = np.asarray(m["task_object_position"])
            nearby = sorted(rows,key=lambda row:np.linalg.norm(np.asarray(row["task_object_position"])-center))[:12]
            scored = []
            for row in nearby:
                state = Path(row["path"])
                assert sha_file(state) == row["sha256"]
                with np.load(state,allow_pickle=False) as z:
                    shifted = z["keypoints"][lo]*(1-blend) + z["keypoints"][hi]*blend
                    score = rms_vector(shifted-sample["decoded_world"])
                scored.append((score,row["sha256"],row))
            seed_error, _, row = min(scored,key=lambda item:item[:2])
            legacy_state = Path(row["path"])
            assert sha_file(legacy_state) == row["sha256"]
            assert sha_file(Path(row["postopt_report"])) == row["postopt_report_sha256"]
            with np.load(legacy_state, allow_pickle=False) as z:
                assert z["joint_names"].tolist() == self.names and float(z["fps"]) == 20.
                seed_q = np.column_stack([np.interp(phase,clock,q) for q in z["q"].T])
                pos = np.column_stack([np.interp(phase,clock,v) for v in z["root_pos"].T])
                rot = Slerp(clock,self.Rotation.from_quat(z["root_quat"][:,[1,2,3,0]]))(phase)
                seed_root = np.c_[rot.as_quat()[:,[3,0,1,2]],pos]
            seed_q = jnp.asarray(np.clip(seed_q,self.low+1e-6,self.high-1e-6), dtype=jnp.float32)
            seed_root = jnp.asarray(seed_root,dtype=jnp.float32)
            report["initialization_only"] = "phase-aligned archive state supplies an optimizer seed; full refit to raw with table collision"
            report["legacy_seed_sha256"] = row["sha256"]
            report["legacy_seed_candidate_id"] = json.loads(Path(row["kinematic_report"]).read_text())["candidate_id"]
            report["legacy_seed_raw_rmse_m"] = seed_error
            qa, ra = self.solve(target, goals, orient, mask, seed_q, seed_root,
                                target, jnp.asarray(initial_q), jnp.asarray(initial_root), ground, stage="A", penalty=0.)
            qa, ra = np.asarray(qa), np.asarray(ra)
            with (output / "stage_a_decoded_realization.npz").open("xb") as f:
                np.savez_compressed(f, q=qa, root=ra, decoded_world=sample["decoded_world"])
            qa_projected, full_a, _, correction_a = self.cpu_state(qa, ra)
            if correction_a >= self.policy.joint_projection_max_rad:
                return finish("generated_candidate_rejected", "stage_a_joint_projection_at_least_1mrad")
            stage_a = {"q": qa_projected, "root": ra, "points": full_a[:, self.kp]}
            report["stage_a_decoded_vector_rmse_m"] = rms_vector(stage_a["points"] - sample["decoded_world"])
            report["stage_a_sha256"] = sha_file(output / "stage_a_decoded_realization.npz")
            chosen = None
            # Each attempt uses the same A anchor and starts from A, avoiding drift
            # through repeated warm starts. No threshold or target is changed.
            for attempt, penalty in enumerate((1., 2., 4.), 1):
                qb, rb = self.solve(target, goals, orient, mask, jnp.asarray(qa_projected), jnp.asarray(ra),
                                    jnp.asarray(stage_a["points"], dtype=jnp.float32), jnp.asarray(qa_projected),
                                    jnp.asarray(ra), ground, stage="B", penalty=penalty)
                qb, rb = np.asarray(qb), np.asarray(rb)
                stage_path = output / ("stage_b_%02d_raw.npz" % attempt)
                with stage_path.open("xb") as f:
                    np.savez_compressed(f, q=qb, root=rb)
                metrics, q, points, wquat = self.measure(sample, qb, rb, stage_a)
                checks = acceptance_checks(metrics, self.policy)
                record = {"attempt": attempt, "penalty_multiplier": penalty, "path": str(stage_path),
                          "sha256": sha_file(stage_path), "metrics": metrics, "checks": checks,
                          "passed": all(checks.values())}
                report["stages"].append(record)
                if [key for key,value in checks.items() if not value] == ["hand_target"]:
                    assert sample["wrist_mask"][:,1].all() and not sample["wrist_mask"][:,0].any()
                    error = sample["targets_world"][:,21]-points[:,21]
                    norm = np.linalg.norm(error,axis=1)
                    radius = self.policy.wrist_position_max_m-.00005
                    scale = np.divide(np.maximum(norm-radius,0.),norm,out=np.zeros_like(norm),where=norm>0.)
                    delta = scale[:,None]*error
                    moved = float(np.linalg.norm(delta,axis=1).max())
                    if moved < .004:
                        projected = rb.copy()
                        projected[:,4:] += delta.astype(rb.dtype)
                        ref_metrics, ref_q, ref_points, ref_wquat = self.measure(sample,qb,projected,stage_a)
                        ref_checks = acceptance_checks(ref_metrics,self.policy)
                        report["stages"].append({"attempt":attempt,"refinement":"minimum norm wrist-ball projection",
                            "max_added_root_translation_m":moved,"metrics":ref_metrics,"checks":ref_checks,
                            "passed":all(ref_checks.values()),"source_sha256":record["sha256"]})
                        if all(ref_checks.values()):
                            record,q,rb,points,wquat = report["stages"][-1],ref_q,projected,ref_points,ref_wquat
                if record["passed"]:
                    chosen = (q, rb, points, wquat, record)
                    break
            if chosen is None:
                return finish("generated_candidate_rejected", "no_bounded_projection_satisfies_all_unchanged_gates")
            q, root, points, wquat, record = chosen
            meta = dict(m, postopt_schema="motion_preserving_postopt_v2", hand_schedule_semantics="planned binary commands; not contact or grasp proof",
                        desired_object_motion_semantics="planning target; not measured dynamics")
            destination = output / "robot29.npz"
            with destination.open("xb") as f:
                np.savez_compressed(f, q=q, root_pos=root[:, 4:], root_quat=root[:, :4],
                    keypoints=points.astype(np.float32), left_wrist_quat=wquat[:, 0].astype(np.float32),
                    right_wrist_quat=wquat[:, 1].astype(np.float32), hand_closed=sample["hand_closed"],
                    planned_object_positions=sample["desired_objects"], desired_wrist_quat_wxyz=sample["orientations"],
                    joint_names=np.array(self.names), fps=np.array(20.), sample_times=np.arange(self.frames) / 20.,
                    task_object_position=np.asarray(m["task_object_position"]), task_object_quaternion_wxyz=np.array([1., 0., 0., 0.]),
                    metadata_utf8=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
            report.update(output=str(destination), output_sha256=sha_file(destination), selected_attempt=record["attempt"],
                          metrics=record["metrics"], checks=record["checks"], passed_requested_kinematic_checks=True,
                          max_hand_error_m=record["metrics"]["wrist_max_l2_m"],
                          max_wrist_angle_rad=record["metrics"]["wrist_max_angle_rad"],
                          max_joint_projection_rad=record["metrics"]["joint_projection_rad"],
                          scope="bounded generated-motion fitting only; independent native scene admission remains required",
                          native_scene_collision_replay="pending")
            return finish("kinematic_candidate_ready_for_native_probe")
        except Exception as e:
            (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            report["passed_requested_kinematic_checks"] = False
            return finish("failed_preserved", type(e).__name__ + ": " + str(e))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--sample-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--fk-module", type=Path, default=Path(__file__).resolve().parents[1] / "g1_fk.py")
    args = p.parse_args()
    sample = load_sample(args.sample, args.sample_sha256)
    solver = MotionPreservingSolverV2(args.urdf, sample["frames"], args.fk_module)
    report = solver.process(args.sample, args.output_dir, args.sample_sha256)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["status"] == "kinematic_candidate_ready_for_native_probe" else 2


if __name__ == "__main__":
    raise SystemExit(main())
