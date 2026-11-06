"""Sample the repaired checkpoint and retain both coordinate frames."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import MethodType

import numpy as np
import torch

from paper_conditions import build
from contracts import GUIDANCE_POLICY, TASK_POINTS, canonical_to_world, coordinate_contract, world_to_canonical
from guidance import attach
from runtime import inputs, load_config, load_legacy, load_task_dependencies, require, sha
from validation import require_receipt


def standing_template(fk, joint_order):
    q = np.zeros((1, 29))
    for side in ("left", "right"):
        for part, value in (("hip_pitch", -.2), ("knee", .42), ("ankle_pitch", -.23),
                            ("elbow", .87), ("shoulder_pitch", .35)):
            q[0, joint_order.index(f"{side}_{part}_joint")] = value
    q[0, joint_order.index("left_shoulder_roll_joint")] = .16
    q[0, joint_order.index("right_shoulder_roll_joint")] = -.16
    points, rotations = fk.forward(np.array([[0., 0., .84]]), np.array([[0., 0., 0., 1.]]),
                                   q, joint_order, return_rotations=True)
    return points[0], rotations[0]


def process(args, cfg, legacy, dependencies, output, checkpoint):
    _, device = legacy.setup({**cfg, "device": cfg.get("sample_device", cfg["device"]),
                              "gpu_memory_fraction": cfg.get("sample_gpu_memory_fraction", .05)})
    urdf, task_sources = load_task_dependencies(cfg)
    task_sources["paper_conditions.py"] = sha(Path(__file__).with_name("paper_conditions.py"))
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from g1_fk import URDFFK
    from model.cfg_sampler import ClassifierFreeSampleModel
    from stage3_pick29_core import BODY_JOINTS, SCENE, SCENE_SHA

    if args.checkpoint_sha256 is None:
        status = json.loads((checkpoint.parent / "status.json").read_text(encoding="utf8"))
        receipt = status["last_checkpoint"]
        require(status["status"] == "completed" and Path(receipt["path"]).resolve() == checkpoint,
                "checkpoint needs the final run receipt or an explicit --checkpoint-sha256")
        args.checkpoint_sha256 = receipt["sha256"]
    digest, receipt = require_receipt(checkpoint)
    require(digest == args.checkpoint_sha256 and receipt["adam_steps"]["cursor_updates"] == cfg["max_steps"],
            "checkpoint digest or exact Adam receipt disagrees with final run")
    data, stats, provenance = inputs(cfg, legacy, dependencies)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["version"] == 2 and saved["provenance"] == provenance
            and saved["config"] == cfg and saved["cursor"]["step"] == cfg["max_steps"],
            "checkpoint is not the bound final repaired-prior checkpoint")
    model, _ = legacy.build_model(cfg)
    legacy.restore_state(model, saved["model"])
    model.to(device)
    model.eval()
    del saved, data
    diffusion = attach(legacy.diffusion_for(stats))
    guide = diffusion.guide
    wrapped = ClassifierFreeSampleModel(model)
    initial, rotations = standing_template(URDFFK(urdf), BODY_JOINTS)
    transform = coordinate_contract(initial, BODY_JOINTS, task_sources["g1_29dof.urdf"], SCENE["ground_z"])
    mean, std = [stats[key].to(device) for key in ("feature_mean", "feature_std")]
    spatial_mean, spatial_std = [stats[key].to(device).reshape(22, 3)
                                 for key in ("spatial_mean", "spatial_std")]
    records = output / "records.jsonl"
    generated = []
    root_status = {"task": args.task, "requested_candidates": args.count, "accepted_references": 0,
                   "checkpoint_sha256": digest, "PPO_started": False}
    with records.open("x", encoding="utf8") as journal:
        for start in range(args.start_index, args.start_index + args.count, args.batch_size):
            floor = cfg["disk_floor_bytes"] + cfg.get("sampling_write_reserve_bytes", 100 * 1024**2)
            require(shutil.disk_usage(output).free > floor, "sampling disk protection")
            ids = list(range(start, min(start + args.batch_size, args.start_index + args.count)))
            batch = len(ids)
            directory = output / f"batch_{start:06d}"
            directory.mkdir()
            condition = build(args.task, ids, args.seed, initial, rotations, SCENE, SCENE_SHA)
            world = condition["world"].astype("<f4")
            canonical = world_to_canonical(world, transform).astype("<f4")
            require(np.allclose(canonical_to_world(canonical, transform), world, atol=1e-6, rtol=0.),
                    "coordinate round-trip exceeded tolerance")
            frames = canonical.shape[1]
            target = torch.as_tensor(canonical, device=device)
            mask = torch.as_tensor(condition["mask"], device=device)
            hint = ((target - spatial_mean) / spatial_std) * mask.unsqueeze(-1)
            caption = condition["prompt"]
            with torch.no_grad():
                embedding = model.encode_text([caption] * batch)
            y = {"text": [caption] * batch, "text_embedding": embedding, "hint": hint.reshape(batch, frames, 66),
                 "hint_mask": mask, "mask": torch.ones((batch, 1, 1, frames), device=device, dtype=torch.bool),
                 "lengths": torch.full((batch,), frames, device=device, dtype=torch.long),
                 "scale": torch.full((batch,), 2.5, device=device)}
            calls = 0

            def record_progress(self, x, t, model_kwargs=None):
                nonlocal calls
                values = guide(x, t, model_kwargs=model_kwargs)
                calls += 1
                if calls % 100 == 0:
                    legacy.write_json(directory / "progress.json", {
                        "status": "sampling", "denoising_steps_completed": calls,
                        "total": diffusion.num_timesteps, "candidates": batch,
                    })
                return values

            diffusion.guide = MethodType(record_progress, diffusion)
            begun = time.monotonic()
            batch_seed = args.seed + 1048576 + start
            torch.manual_seed(batch_seed)
            noise = torch.cat([torch.randn((1, 263, 1, frames), device=device,
                                           generator=torch.Generator(device=device).manual_seed(args.seed + i))
                               for i in ids])
            with torch.no_grad():
                normalized = diffusion.p_sample_loop(wrapped, (batch, 263, 1, frames), noise=noise,
                                                     clip_denoised=False, model_kwargs={"y": y},
                                                     device=device, progress=False)
                features = normalized[:, :, 0].transpose(1, 2) * std + mean
                points = recover_from_ric(features, 22)
            require(calls == diffusion.num_timesteps, "reverse loop failed to invoke spatial guidance each step")
            require(torch.isfinite(features).all() and torch.isfinite(points).all(), "nonfinite sampled motion")
            features, points, noise = (tensor.cpu().numpy() for tensor in (features, points, noise))
            rows = []
            for k, candidate_id in enumerate(ids):
                path = directory / f"sample_{k}.npz"
                grasp_frame = condition["grasp_frames"][k]
                error = np.linalg.norm(points[k, :, 21] - canonical[k, :, 21], axis=-1)
                noise_digest = hashlib.sha256(noise[k].tobytes()).hexdigest()
                feature_digest = hashlib.sha256(features[k].tobytes()).hexdigest()
                world_digest = hashlib.sha256(world[k].tobytes()).hexdigest()
                planned_object = condition["object_initial"][k].tolist()
                metadata = {
                    **transform, "caption": caption, "fps": 20, "frames": frames, "task": args.task,
                    "checkpoint_sha256": digest, "candidate_id": candidate_id, "initial_noise_seed": args.seed + candidate_id,
                    "denoising_batch_seed": batch_seed, "denoising_batch_ids": ids, "initial_noise_sha256": noise_digest,
                    "feature_content_sha256": feature_digest, "target_world_content_sha256": world_digest,
                    "task_object_position": planned_object, "task_plan": condition["plans"][k],
                    "scene_sha256": condition["scene_sha256"], "grasp_frame": grasp_frame, "grasp_time_s": grasp_frame / 20.,
                    "physical_acceptance": False, "guidance": GUIDANCE_POLICY, "sampling_sources": task_sources,
                    "requested_constraint_points": sorted(set(np.flatnonzero(condition["mask"][k].any(0)).tolist())),
                    "task_conditions_source_sha256": task_sources["paper_conditions.py"],
                }
                with path.open("xb") as file:
                    np.savez_compressed(file, features=features[k:k+1], canonical_points=points[k:k+1],
                                        target_points=canonical[k], target_points_world=world[k],
                                        target_mask=condition["mask"][k],
                                        desired_object_positions=condition["desired_object_positions"][k],
                                        desired_wrist_quat_wxyz=condition["desired_wrist_quat_wxyz"][k],
                                        planned_hand_closed=condition["planned_hand_closed"][k],
                                        metadata_utf8=np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8))
                row = {"path": str(path), "sha256": sha(path), "candidate_id": candidate_id, "task": args.task,
                       "initial_noise_sha256": noise_digest, "feature_content_sha256": feature_digest,
                       "task_object_position": planned_object, "grasp_frame": grasp_frame,
                       "wrist_error_at_grasp_m": float(error[grasp_frame]),
                       "max_wrist_error_m": float(error.max()), "mean_wrist_error_m": float(error.mean())}
                rows.append(row)
            legacy.write_json(directory / "report.json", {**root_status, "status": "candidates_generated",
                              "candidates": rows, "elapsed_s": time.monotonic() - begun}, exclusive=True)
            for row in rows:
                journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            generated.extend(rows)
            legacy.write_json(output / "status.json", {**root_status, "status": "sampling",
                              "generated_candidates": len(generated), "latest_batch": start})
    require(len(generated) == args.count, "candidate count mismatch")
    for key in ("candidate_id", "initial_noise_sha256", "feature_content_sha256"):
        require(len({row[key] for row in generated}) == args.count, f"duplicate candidate {key}")
    final = {**root_status, "status": "sampling_completed_not_postoptimized", "generated_candidates": len(generated),
             "records_sha256": sha(records), "coordinate_transform": transform, "sampling_sources": task_sources}
    legacy.write_json(output / "report.json", final, exclusive=True)
    legacy.write_json(output / "status.json", final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_POINTS), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve(strict=True)
    cfg = load_config(args.config)
    require(1 <= args.count <= 1000 and 1 <= args.batch_size <= cfg["max_sample_batch_size"]
            and args.start_index >= 0 and args.seed >= 0, "invalid sample count, ID range, seed or batch size")
    require(output.is_relative_to(Path(cfg["stage_dir"])), "sample into the isolated repaired stage")
    legacy, dependencies = load_legacy(cfg)
    output.mkdir(exist_ok=False, parents=True)
    legacy.write_json(output / "status.json", {"status": "initializing", "task": args.task,
                      "requested_candidates": args.count, "generated_candidates": 0, "PPO_started": False})
    try:
        process(args, cfg, legacy, dependencies, output, checkpoint)
    except Exception as error:
        legacy.write_json(output / "failure.json", {"type": type(error).__name__, "error": str(error)}, exclusive=True)
        raise


if __name__ == "__main__":
    main()

