"""Paper V-C controls, with explicit 29DoF tabletop implementation choices.

The paper does not provide a full Pour angle curve or alpha_l. We use alpha_l=1
and retain the documented -75 degree task angle. The phase clock is shared by
position, elbow, wrist orientation, hand commands and planned object poses.
"""
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from conditions import build as previous_build

PROTOCOL = "paper_conditions_and_filtering_v7"
PAPER_URL = "https://arxiv.org/html/2604.00202v1#S5.SS3"


def elbow_positions(wrist, initial, rotations):
    """Side-grasp elbow hint, smoothly rotated with the pouring wrist."""
    length = np.linalg.norm(initial[21] - initial[19])
    # Apply the SAME proper rotation as the position adapter to the paper's
    # [-alpha_l*l*cos(theta), 0, -l*sin(theta)] offset (alpha_l=1).
    base_offset = np.array([0., 1., -1.]) * length / np.sqrt(2.)
    grasp = Rotation.from_euler("y", 90., degrees=True)
    offset = (rotations * grasp.inv()).apply(np.broadcast_to(base_offset, wrist.shape))
    return wrist + offset


def build(task, candidate_ids, seed, initial, rotations, scene, scene_sha):
    result = previous_build(task, candidate_ids, seed, initial, rotations, scene, scene_sha)
    if task not in ("pick", "pour"):
        for plan in result["plans"]:
            plan["protocol"] = PROTOCOL
            plan["paper_task_equivalence"] = "project adaptation; no identical published task recipe"
        return result

    n = result["world"].shape[1]
    time = np.arange(n) / 20.
    prompt = ("a person picks an object with their right hand" if task == "pick" else
              "a person picks an object with the right hand and performs a pouring action with the same hand")
    result["prompt"] = prompt
    result["mask"][:] = False
    result["mask"][:, :, [19, 21]] = True
    result["mask"][:, ::5, [4, 5]] = True
    result["grasp_frames"] = []
    for b, candidate in enumerate(candidate_ids):
        rng = np.random.default_rng(seed + candidate)
        lateral = rng.uniform(-.22, .12) if task == "pick" else rng.uniform(-.10, .10)
        forward = rng.uniform(.2, .4)
        # Paper prompt lateral/forward/Z is converted to the project's X-forward,
        # Y-left, Z-up world. This explicit frame adaptation is retained in metadata.
        center = np.array([forward, -lateral, .90])
        frame = int(rng.integers(40, 51 if task == "pick" else 43))
        contact = frame / 20.
        result["grasp_frames"].append(frame)
        if task == "pour":
            knots = np.array([0., contact, contact + 1.5, contact + 2.5,
                              contact + 3.5, contact + 5., contact + 6., time[-1]])
            positions = np.array([center, center, center + [0., .08, .16],
                                  center + [0., .08, .16], center + [0., .08, .16],
                                  center + [0., 0., .16], center, center])
            angles = np.interp(time, [0., contact, contact + 1.5, contact + 2.5,
                                     contact + 3.5, contact + 4.5, time[-1]],
                               [0., 0., 0., -75., -75., 0., 0.])
            desired_rotation = Rotation.from_euler("x", angles, degrees=True) * Rotation.from_euler("y", 90., degrees=True)
            closed = (time >= contact) & (time < contact + 6.)
        else:
            knots = np.array([0., contact, contact + 2., time[-1]])
            positions = np.array([center, center, center + [0., 0., .16], center + [0., 0., .16]])
            desired_rotation = Rotation.from_euler("y", np.full(n, 90.), degrees=True)
            closed = time >= contact
        obj = np.column_stack([np.interp(time, knots, positions[:, d]) for d in range(3)])
        wrist = obj + [0., 0., .04]
        before = time <= contact
        blend = (time[before] / contact)[:, None]
        wrist[before] = initial[21] * (1. - blend) + wrist[frame] * blend
        approach = Slerp([0., contact], Rotation.from_matrix(np.stack([rotations[21], desired_rotation[frame].as_matrix()])))(np.minimum(time, contact))
        quat = desired_rotation.as_quat()
        quat[before] = approach.as_quat()[before]
        desired_rotation = Rotation.from_quat(quat)
        elbow = elbow_positions(wrist, initial, desired_rotation)
        elbow[before] = initial[19] * (1. - blend) + elbow[frame] * blend
        result["world"][b, :, 21] = wrist
        result["world"][b, :, 19] = elbow
        result["object_initial"][b] = center
        result["desired_object_positions"][b] = obj
        result["desired_wrist_quat_wxyz"][b, :, 1] = quat[:, [3, 0, 1, 2]]
        result["planned_hand_closed"][b, :, 1] = closed
        result["plans"][b].update(protocol=PROTOCOL, source_caption=prompt,
            object_initial_position=center.tolist(),
            paper_control_position=[float(lateral), float(forward), .94],
            paper_to_project_position="X=paper_y,Y=-paper_x,Z=paper_z (explicit scene-frame adaptation)",
            elbow_alpha_l=1., elbow_theta_rad=float(np.pi / 4),
            elbow_rotation_rule="rotate forearm hint with desired wrist tilt after contact",
            implementation_choices="29DoF; paper lateral/forward frame adapter; alpha_l=1; existing Pour -75deg curve",
            phases={"contact_s":contact,"lift_s":contact+1.5 if task == "pour" else contact+2.,
                    "tilt_s":contact+2.5 if task == "pour" else None,
                    "tilt_hold_end_s":contact+3.5 if task == "pour" else None,
                    "upright_s":contact+4.5 if task == "pour" else None,
                    "place_s":contact+6. if task == "pour" else None})
    return result


def augment_existing_sample(values):
    """Apply elbow/wrist coupling to a preserved sample for solver validation.

    Original decoded motion, wrist targets, object plan and actual generation
    caption are retained. This validates postoptimization only, not new sampling.
    """
    result = {key: np.array(value, copy=True) for key, value in values.items()}
    m = json.loads(result["metadata_utf8"].tobytes())
    world = result["target_points_world"].astype(float)
    orientation = Rotation.from_quat(result["desired_wrist_quat_wxyz"][:, 1][:, [1, 2, 3, 0]])
    elbow = elbow_positions(world[:, 21], world[0], orientation)
    frame = int(m["grasp_frame"])
    weight = (np.arange(frame + 1) / frame)[:, None]
    elbow[:frame+1] = world[0, 19] * (1. - weight) + elbow[frame] * weight
    world[:, 19] = elbow
    result["target_points_world"] = world.astype(np.float32)
    from contracts import world_to_canonical
    result["target_points"] = world_to_canonical(world, m).astype(np.float32)
    result["target_mask"][:, 19] = True
    m["original_target_world_content_sha256"] = m["target_world_content_sha256"]
    m["target_world_content_sha256"] = hashlib.sha256(result["target_points_world"].tobytes()).hexdigest()
    m.update(postopt_control_protocol=PROTOCOL, actual_diffusion_elbow_hint=False,
             existing_sample_validation_only=True)
    result["metadata_utf8"] = np.frombuffer(json.dumps(m).encode(), dtype=np.uint8)
    return result
