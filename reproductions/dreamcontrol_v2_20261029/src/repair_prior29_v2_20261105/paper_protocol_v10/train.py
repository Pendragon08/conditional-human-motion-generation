"""Train the separate 29DoF robot prior for 10,000 optimizer updates."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from contracts import TASK_POINTS, task_mask
from guidance import attach
from runtime import inputs, load_config, load_legacy, require
from validation import buffer_witness, require_receipt, wrap_checkpoint


def batch_from(data, stats, embeddings, device, *, training=True, generator=None):
    n = max(len(row["motion"]) for row in data)
    motion = torch.zeros((len(data), 263, 1, n), device=device)
    hint = torch.zeros((len(data), n, 22, 3), device=device)
    valid = torch.zeros((len(data), n), dtype=torch.bool, device=device)
    mask = torch.zeros_like(hint[..., 0], dtype=torch.bool)
    spatial_mean = stats["spatial_mean"].to(device).reshape(22, 3)
    spatial_std = stats["spatial_std"].to(device).reshape(22, 3)
    lengths, captions = [], []
    profiles = tuple(TASK_POINTS)
    for i, row in enumerate(data):
        length = len(row["motion"])
        lengths.append(length)
        valid[i, :length] = True
        motion[i, :, 0, :length] = row["motion"].T.to(device)
        profile = profiles[int(torch.randint(len(profiles), (), generator=generator))]
        if not training or random.random() >= .10:
            mask[i, :length] = torch.as_tensor(task_mask(profile, length), device=device)
        hint[i, :length] = (row["points"].to(device) - spatial_mean) / spatial_std
        captions.append(random.choice(row["captions"]) if training else row["captions"][0])
    hint *= mask.unsqueeze(-1)
    y = {"mask": valid[:, None, None], "hint_mask": mask, "hint": hint.reshape(len(data), n, 66),
         "lengths": torch.as_tensor(lengths, device=device), "text": captions,
         "text_embedding": torch.stack([embeddings[caption] for caption in captions])}
    return motion, {"y": y}


def loss_for(model, diffusion, motion, condition, timesteps=None, noise=None):
    if timesteps is None:
        timesteps = torch.randint(diffusion.num_timesteps, (len(motion),), device=motion.device)
    if noise is None:
        noise = torch.randn_like(motion)
    noisy = diffusion.q_sample(motion, timesteps, noise=noise)
    guided = diffusion.guide(noisy, timesteps, model_kwargs=condition, train=True)
    prediction = model(guided, timesteps, **condition)
    mask = condition["y"]["mask"]
    per_sample = ((prediction - motion).square() * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * 263)
    require(torch.isfinite(per_sample).all(), "nonfinite denoising loss")
    return per_sample.mean()


def reverse_step_witness(model, diffusion, item, stats, embeddings, device):
    """Exercise the true CFG reverse step and full early/late guidance branches."""
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    from model.cfg_sampler import ClassifierFreeSampleModel

    truncated = {**item, "motion": item["motion"][:60], "points": item["points"][:60]}
    model.eval()
    motion, condition = batch_from([truncated], stats, embeddings, device, training=False,
                                   generator=torch.Generator().manual_seed(41))
    condition["y"]["scale"] = torch.full((1,), 2.5, device=device)
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(123)
        with torch.no_grad():
            wrapped = ClassifierFreeSampleModel(model)
            outputs = [diffusion.p_sample(wrapped, motion, torch.tensor([t], device=device),
                                          clip_denoised=False, model_kwargs=condition)["sample"]
                       for t in (50, 0)]
            require(all(torch.isfinite(x).all() for x in outputs), "reverse step witness failed")
            features = outputs[-1][:, :, 0].transpose(1, 2)
            features = features * stats["feature_std"].to(device) + stats["feature_mean"].to(device)
            points = recover_from_ric(features, 22)
            require(torch.isfinite(points).all(), "nonfinite witness keypoints")
    mask = condition["y"]["hint_mask"]
    return {"features": features.cpu().numpy(), "canonical_points": points.cpu().numpy(),
            "target_points": truncated["points"].cpu().numpy(), "target_mask": mask.cpu().numpy(),
            "metadata": {"scope": "actual CUDA reverse-step witness; no full candidate sampled",
                         "timesteps": [50, 0], "guidance_iterations": [10, 500],
                         "caption": condition["y"]["text"][0], "frames": len(truncated["motion"])}}


def smoke(cfg, legacy, data, stats, provenance, device):
    model, initialization = legacy.build_model(cfg)
    model.to(device)
    embeddings = legacy.text_cache(model, data, device)
    diffusion = legacy.diffusion_for(stats)
    training = sorted((row for row in data if row["row"]["split"] == "train"),
                      key=lambda row: len(row["motion"]))
    motion, condition = batch_from(training[:4] + training[-4:], stats, embeddings, device)
    valid = condition["y"]["mask"]
    t = torch.full((len(motion),), 500, device=device, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        original = model(motion, t, **condition)
        changed = motion.masked_fill(~valid, 1234.)
        padded = model(changed, t, **condition)
        padding_delta = float(((original - padded) * valid).abs().max())
    require(padding_delta < 1e-5, "padded values affected valid model frames")

    initial_zero = model.zero_convs[0].weight.detach().clone()
    initial_hint = model.input_hint_block.poseEmbedding[-1].weight.detach().clone()
    model.train()
    model.clip_model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    losses, seconds = [], []
    for _ in range(8):
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, diffusion, motion, condition)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        require(torch.isfinite(norm), "nonfinite gradients")
        optimizer.step()
        torch.cuda.synchronize(device)
        losses.append(float(loss))
        seconds.append(time.monotonic() - began)
    zero_delta = float((model.zero_convs[0].weight - initial_zero).abs().max())
    hint_delta = float((model.input_hint_block.poseEmbedding[-1].weight - initial_hint).abs().max())
    require(zero_delta > 0 and hint_delta > 0, "control and hint networks did not train")
    cursor = {"step": 8, "epoch": 0, "next_batch": 0}
    payload = legacy.payload(model, optimizer, cursor, cfg, provenance, device)
    size, validation = buffer_witness(payload, cfg["checkpoint_size_budget"])
    del payload
    witness = next(row for row in data if row["row"]["split"] == "val"
                   and row["row"]["dataset_source"] == "OmniRetarget/OMOMO")
    sampled = reverse_step_witness(model, diffusion, witness, stats, embeddings, device)
    out = Path(cfg["stage_dir"]) / "reports"
    with (out / "smoke_sample.npz").open("xb") as file:
        np.savez_compressed(file, features=sampled["features"], canonical_points=sampled["canonical_points"],
                            target_points=sampled["target_points"], target_mask=sampled["target_mask"],
                            metadata_utf8=np.frombuffer(json.dumps(sampled["metadata"]).encode(), dtype=np.uint8))
    report = {"model_initialization": initialization, "provenance": provenance,
              "real_gpu_optimizer_steps": validation["adam_steps"]["max_parameter_updates"],
              "losses": losses, "microbatch_size": len(motion), "steady_microbatch_seconds": sum(seconds[2:]) / 6,
              "valid_frame_padding_max_difference": padding_delta,
              "zero_conv_weight_change": zero_delta, "hint_weight_change": hint_delta,
              "real_adam_checkpoint_serialized_bytes": size, "checkpoint_roundtrip": True,
              "checkpoint_validation": validation, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "sampling": sampled["metadata"], "scope": "eight real updates and full checkpoint witness"}
    legacy.write_json(out / "gpu_smoke.json", report, exclusive=True)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("verify-env", "preflight", "smoke", "train"), required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    require(args.resume is None or args.mode == "train", "resume is for train mode")
    resume = args.resume.resolve(strict=True) if args.resume else None
    if resume:
        require_receipt(resume)
    cfg = load_config(args.config)
    legacy, dependencies = load_legacy(cfg)
    _, device = legacy.setup(cfg)
    if args.mode == "verify-env":
        gen = torch.Generator(device=device).manual_seed(17)
        x = torch.randn((16, 16), generator=gen, device=device)
        out = x @ x.T
        require(torch.isfinite(out).all(), "CUDA kernel witness failed")
        print(json.dumps({"device": str(device), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
                          "seed": 17, "checksum": float(out.sum()), "sources": dependencies}), flush=True)
        return
    data, stats, provenance = inputs(cfg, legacy, dependencies)
    reports = Path(cfg["stage_dir"]) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        legacy.write_json(reports / "preflight.json", provenance, exclusive=True)
        print(json.dumps(provenance), flush=True)
        return
    # Override callbacks on this isolated module object only. Original files,
    # module imports, model tensors and trainer checkpoint machinery stay bound.
    upstream_diffusion = legacy.diffusion_for
    legacy.diffusion_for = lambda stats: attach(upstream_diffusion(stats))
    legacy.batch_from = batch_from
    legacy.loss_for = loss_for
    legacy.checkpoint = wrap_checkpoint(legacy.checkpoint, legacy.write_json)
    if args.mode == "smoke":
        smoke(cfg, legacy, data, stats, provenance, device)
    else:
        witness = json.loads((reports / "gpu_smoke.json").read_text(encoding="utf8"))
        require(witness["provenance"] == provenance and witness["checkpoint_validation"]["all_fields_exact"] is True,
                "repair GPU witness source/data binding or full checkpoint verification missing")
        legacy.train(cfg, data, stats, provenance, device, resume)


if __name__ == "__main__":
    main()

