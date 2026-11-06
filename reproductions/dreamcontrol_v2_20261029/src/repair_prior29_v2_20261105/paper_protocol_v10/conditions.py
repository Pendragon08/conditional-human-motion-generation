"""Build world-frame controls before applying the shared model transform."""
import numpy as np

from contracts import task_mask


def build(task, candidate_ids, seed, initial, rotations, scene, scene_sha):
    if task != "pick":
        from stage3_supported_task_conditions import build as supported
        result = supported(task, candidate_ids, seed, initial, rotations)
        result["grasp_frames"] = [40] * len(candidate_ids)
    else:
        result = _pick(candidate_ids, seed, initial, rotations, scene, scene_sha)
    frames = result["world"].shape[1]
    result["mask"] = np.broadcast_to(task_mask(task, frames),
                                     (len(candidate_ids), frames, 22)).copy()
    return result


def _pick(candidate_ids, seed, initial, rotations, scene, scene_sha):
    from scipy.spatial.transform import Rotation, Slerp

    frames = 120
    times = np.arange(frames) / 20.
    world = np.broadcast_to(initial, (len(candidate_ids), frames, 22, 3)).copy()
    prompt = "a person picks an object with their right hand"
    objects, poses, closed, object_paths, plans, grasp_frames = [], [], [], [], [], []
    forearm = np.linalg.norm(initial[21] - initial[19])
    elbow_offset = np.array([-1., 0., -1.]) * (forearm / np.sqrt(2.))
    right_grasp = Rotation.from_euler("y", 90., degrees=True)
    left_initial = Rotation.from_matrix(rotations[20]).as_quat()
    right_initial = Rotation.from_matrix(rotations[21]).as_quat()
    for batch_index, candidate_id in enumerate(candidate_ids):
        rng = np.random.default_rng(seed + candidate_id)
        center = np.asarray(scene["object_center"]) + np.r_[rng.uniform(-.06, .06, 2), 0.]
        grasp_frame = int(rng.integers(40, 51))
        grasp_frames.append(grasp_frame)
        contact = grasp_frame / 20.
        lift_end = contact + 2.
        right_target = center + [0., 0., .04]
        right_lift = center + [0., 0., .20]
        knots = [0., contact, lift_end]
        for point, offset in ((21, np.zeros(3)), (19, elbow_offset)):
            positions = np.array([initial[point], right_target + offset, right_lift + offset])
            world[batch_index, :, point] = np.stack(
                [np.interp(times, knots, positions[:, dim]) for dim in range(3)], axis=-1
            )
        right = Slerp([0., max(contact - .5, .01)], Rotation.from_quat(
            np.stack((right_initial, right_grasp.as_quat()))
        ))(np.minimum(times, contact - .5)).as_quat()
        left = np.broadcast_to(left_initial, right.shape)
        poses.append(np.stack((left[:, [3, 0, 1, 2]], right[:, [3, 0, 1, 2]]), axis=1))
        closed.append(np.column_stack((np.zeros(frames, dtype=np.int8), times >= contact)))
        objects.append(center)
        object_paths.append(np.repeat(center[None], frames, axis=0)
                            + np.column_stack((np.zeros((frames, 2)),
                                               np.interp(times, knots, [0., 0., .16]))))
        plans.append({"task": "pick", "candidate_id": candidate_id, "scene_sha256": scene_sha,
                      "phases": {"contact_s": contact, "lift_s": lift_end},
                      "object_initial_position": center.tolist(),
                      "position_targets_enter_diffusion": True,
                      "orientation_targets": "postoptimization_only",
                      "hand_schedule": "planned binary commands",
                      "source_caption": prompt, "final_reference_admission": False})
    return {
        "world": world, "object_initial": np.asarray(objects),
        "desired_object_positions": np.asarray(object_paths),
        "desired_wrist_quat_wxyz": np.asarray(poses),
        "planned_hand_closed": np.asarray(closed, dtype=np.int8),
        "plans": plans, "scene": scene, "scene_sha256": scene_sha, "prompt": prompt,
        "grasp_frames": grasp_frames,
    }

