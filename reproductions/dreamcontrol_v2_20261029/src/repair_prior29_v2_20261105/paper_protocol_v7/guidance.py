"""Explicit-mask port of the fixed DreamControl's spatial guide.

Keep its summed point L2, posterior variance floor, per-density sampling scale,
and iteration schedule. Apply the train-time timestep branch per item in a mixed
batch. Empty hints and padding never contribute. No gradient goes back through
the q_sample correction or reverse mean correction.
"""
from types import MethodType

from contracts import GUIDANCE_POLICY


def iteration_counts(timesteps, training):
    policy = GUIDANCE_POLICY["train" if training else "sample"]
    threshold = 20 if training else 10
    return tuple(policy[f"below_{threshold}_steps"] if t < threshold else policy["steps"]
                 for t in timesteps)


def attach(diffusion):
    import torch
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    def guide(self, x, t, model_kwargs=None, *, train=False):
        y = model_kwargs["y"]
        valid = y["mask"][:, 0, 0]
        mask = y["hint_mask"] & valid.unsqueeze(-1)
        active = mask.any(dim=(1, 2))
        if not active.any():
            return x.detach()
        stats = [item.to(device=x.device, dtype=x.dtype)
                 for item in (self.mean, self.std, self.raw_mean, self.raw_std)]
        feature_mean, feature_std, spatial_mean, spatial_std = stats
        target = (y["hint"] * spatial_std + spatial_mean).reshape(*mask.shape, 3)
        logvar = torch.as_tensor(self.posterior_log_variance_clipped, device=x.device, dtype=x.dtype)
        variance = logvar[t].exp().clamp_min(GUIDANCE_POLICY["posterior_variance_floor"])
        schedule = iteration_counts(t.detach().tolist(), train)
        counts = torch.tensor(schedule, device=x.device)
        result = x.detach().clone()
        for steps in sorted(set(schedule)):
            indices = torch.where(active & (counts == steps))[0]
            if not indices.numel():
                continue
            points_mask = mask[indices].unsqueeze(-1)
            goals = target[indices]
            if train:
                scale = GUIDANCE_POLICY["train"]["scale"]
            else:
                density = mask[indices].sum(dim=1).amax(dim=1).to(x.dtype)
                scale = GUIDANCE_POLICY["sample"]["density_scale"] / density
            gain = (variance[indices] * scale).reshape(-1, 1, 1, 1)
            values = result[indices]
            for _ in range(steps):
                with torch.enable_grad():
                    work = values.detach().requires_grad_(True)
                    features = work[:, :, 0].transpose(1, 2) * feature_std + feature_mean
                    points = recover_from_ric(features, 22)
                    residual = (points - goals).masked_fill(~points_mask, 0)
                    loss = torch.linalg.vector_norm(residual, dim=-1).sum()
                    (gradient,) = torch.autograd.grad(loss, work)
                gradient[..., 0] = 0
                values = work.detach() - gain * gradient
            result[indices] = values
        if not torch.isfinite(result).all():
            raise FloatingPointError("non-finite result from spatial guidance")
        return result

    diffusion.guide = MethodType(guide, diffusion)
    return diffusion

