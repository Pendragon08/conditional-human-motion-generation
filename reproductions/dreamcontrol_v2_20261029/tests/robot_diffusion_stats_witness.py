"""Exercise upstream guidance with synthetic robot statistics on CPU.

This is not training; synthetic data do not establish robotics validity.
"""
import json
import numpy as np
import torch
from diffusion import gaussian_diffusion as gd
from robot_diffusion_stats import configure_robot_stats


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    diffusion = gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('cosine', 1000, 1.),
                                    model_mean_type=gd.ModelMeanType.START_X,
                                    model_var_type=gd.ModelVarType.FIXED_SMALL,
                                    loss_type=gd.LossType.MSE, dataset='humanml')
    configure_robot_stats(diffusion, np.zeros(263), np.ones(263), np.zeros(66), np.ones(66))
    x = torch.randn(1, 263, 1, 20) * .05
    t = torch.tensor([500])
    empty = torch.zeros(1, 20, 66)
    targeted = empty.clone()
    targeted[0, 10, 60] = .3
    clean = diffusion.guide(x.clone(), t, model_kwargs={'y': {'hint': empty}}, train=False)
    guided = diffusion.guide(x.clone(), t, model_kwargs={'y': {'hint': targeted}}, train=False)
    print(json.dumps({'frames': 20, 'zero_hint_max_change': float((clean-x).abs().max()),
                      'nonzero_hint_change_l2': float(torch.linalg.vector_norm(guided-x)),
                      'finite': bool(torch.isfinite(guided).all()),
                      'caveat': 'Synthetic guidance only; upstream hint-mask and multi-timestep semantics not fixed.'}))


if __name__ == '__main__':
    main()
