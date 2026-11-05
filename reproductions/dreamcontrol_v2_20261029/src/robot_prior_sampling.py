"""Experimental text/spatial sampling using the original DDPM and CFG code.

The caller loads a checkpoint and configures robot statistics.  Spatial
guidance uses this project's explicit-mask MSE approximation, not verified
DreamControl-v2 author code.  Outputs are canonical 22-point coordinates and
263D features, not yet joint-space trajectories or a trained policy.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
from types import MethodType
import torch
from robot_spatial_guidance import guide as explicit_guide


@contextmanager
def _explicit_sampling_guide(diffusion, *, step_size, steps):
    """Temporarily replace this instance's implicit-mask sampler guidance."""
    if getattr(diffusion, '_robot_sampling_active', False):
        raise RuntimeError('concurrent sampling on one diffusion instance is unsupported')
    previous = diffusion.__dict__.get('guide')
    had_instance_guide = 'guide' in diffusion.__dict__
    def adapter(self, x, t, model_kwargs=None, **unused):
        if not isinstance(model_kwargs, dict) or not isinstance(model_kwargs.get('y'), dict):
            raise ValueError('explicit sampling conditions required')
        y = model_kwargs['y']
        return explicit_guide(self, x, t, y['hint'], y['hint_mask'], step_size=step_size,
                              steps=steps, valid_frames=y['mask'][:, 0, 0, :])
    diffusion._robot_sampling_active = True
    diffusion.guide = MethodType(adapter, diffusion)
    try:
        yield
    finally:
        if had_instance_guide:
            diffusion.guide = previous
        else:
            del diffusion.guide
        del diffusion._robot_sampling_active


def sample_robot_prior(model, diffusion, *, texts, targets, target_mask,
                       cfg_scale, spatial_step_size, spatial_steps, seed):
    """Sample equal-length canonical motions from an already-loaded model.

    ``targets`` is BxTx22x3 in the canonical coordinates decoded from the
    project's 263D representation, not an Isaac/world-space position.  The
    separate boolean mask allows constrained coordinates whose value is zero.
    A caller must do frame transforms and validate checkpoint/data provenance.
    The MSE guidance freezes all feature gradients at frame zero, following the
    current experimental guide; first-frame goals are not directly optimized.
    """
    from model.cfg_sampler import ClassifierFreeSampleModel
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    if not isinstance(texts, list) or not texts or any(not isinstance(s, str) or not s.strip() for s in texts):
        raise ValueError('one nonempty text string per sample is required')
    if not isinstance(targets, torch.Tensor) or targets.ndim != 4 or targets.shape[0] != len(texts) or targets.shape[2:] != (22, 3):
        raise ValueError('targets must have shape (B,T,22,3)')
    if not torch.is_floating_point(targets) or not torch.isfinite(targets).all() or not 2 <= targets.shape[1] <= 199:
        raise ValueError('targets must be finite, floating point and have 2–199 frames')
    if not isinstance(target_mask, torch.Tensor) or target_mask.shape != targets.shape[:-1] or target_mask.dtype != torch.bool or target_mask.device != targets.device:
        raise ValueError('target mask must be boolean (B,T,22) on the target device')
    if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63 or
        isinstance(spatial_steps, bool) or not isinstance(spatial_steps, int) or spatial_steps < 0 or
        isinstance(cfg_scale, bool) or not isinstance(cfg_scale, (int, float)) or not math.isfinite(cfg_scale) or cfg_scale < 0 or
        isinstance(spatial_step_size, bool) or not isinstance(spatial_step_size, (int, float)) or not math.isfinite(spatial_step_size) or spatial_step_size < 0):
        raise ValueError('invalid random seed or guidance settings')
    if model.training or getattr(model, 'njoints', None) != 263 or getattr(model, 'nfeats', None) != 1 or getattr(model, 'cond_mask_prob', 0) <= 0:
        raise ValueError('expected eval-mode 263D CFG-capable CMDM')
    try:
        param = next(model.parameters())
    except StopIteration as e:
        raise ValueError('model has no parameters') from e
    if param.device != targets.device or param.dtype != torch.float32 or targets.dtype != torch.float32:
        raise ValueError('model and targets must share a device and use float32')
    if not hasattr(model, 'hint_mask_embedding'):
        raise ValueError('model must have the explicit hint-mask branch attached')
    stats=[]
    for name, size in [('mean',263),('std',263),('raw_mean',66),('raw_std',66)]:
        value=getattr(diffusion,name,None)
        if not isinstance(value,torch.Tensor) or value.numel()!=size or not torch.isfinite(value).all():
            raise ValueError('diffusion robot statistics are missing or invalid')
        stats.append(value.to(device=targets.device,dtype=targets.dtype).reshape(-1))
    mean,std,spatial_mean,spatial_std=stats
    if torch.any(std<=0) or torch.any(spatial_std<=0):raise ValueError('standard deviations must be positive')
    batch,frames=targets.shape[:2]
    normalized=((targets.reshape(batch,frames,66)-spatial_mean)/spatial_std).reshape(batch,frames,22,3)
    hint=(normalized*target_mask.unsqueeze(-1)).reshape(batch,frames,66)
    y={'text':texts,'hint':hint,'hint_mask':target_mask,
       'mask':torch.ones((batch,1,1,frames),device=targets.device,dtype=torch.bool),
       'lengths':torch.full((batch,),frames,device=targets.device,dtype=torch.long),
       'scale':torch.full((batch,),float(cfg_scale),device=targets.device,dtype=targets.dtype)}
    wrapped=ClassifierFreeSampleModel(model)
    device_index = targets.device.index if targets.device.index is not None else (torch.cuda.current_device() if targets.device.type=='cuda' else None)
    cuda_devices=[device_index] if targets.device.type=='cuda' else []
    with torch.random.fork_rng(devices=cuda_devices), _explicit_sampling_guide(diffusion,step_size=spatial_step_size,steps=spatial_steps):
        torch.random.default_generator.manual_seed(seed)
        if cuda_devices:
            torch.cuda.default_generators[device_index].manual_seed(seed)
        noise=torch.randn((batch,263,1,frames),device=targets.device,dtype=targets.dtype)
        sampled=diffusion.p_sample_loop(wrapped,(batch,263,1,frames),noise=noise,clip_denoised=False,
                                        model_kwargs={'y':y},device=targets.device,progress=False)
    if not isinstance(sampled,torch.Tensor) or sampled.shape!=(batch,263,1,frames) or not torch.isfinite(sampled).all():
        raise ValueError('diffusion returned invalid samples')
    features=sampled.permute(0,3,2,1).squeeze(2)*std+mean
    points=recover_from_ric(features,22)
    if points.shape!=(batch,frames,22,3) or not torch.isfinite(points).all():raise ValueError('invalid decoded points')
    return {'normalized_features':sampled.detach(),'canonical_points':points.detach(),
            'metadata':{'seed':seed,'cfg_scale':float(cfg_scale),'spatial_step_size':float(spatial_step_size),
                        'spatial_steps':spatial_steps,'diffusion_steps':int(diffusion.num_timesteps),
                        'frames':frames,'target_frame':'canonical 263D decoder coordinates',
                        'guidance':'explicit-mask MSE approximation; not author v2 code',
                        'mse_guidance_first_frame_feature_gradient_frozen':True,
                        'training_or_physical_acceptance':False}}
