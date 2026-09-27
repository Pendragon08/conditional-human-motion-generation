"""Seeded GPU-kernel and official PyTorch3D rotation witness, without data assets."""

import numpy as np
import torch
from pytorch3d import transforms


torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
device = torch.device("cuda:0")
x = torch.randn((8, 8), device=device)
y = x @ x.T
angles = torch.tensor([[0.1, 0.2, -0.3]], device=device)
quaternion = transforms.matrix_to_quaternion(
    transforms.euler_angles_to_matrix(angles, convention="XYZ")
)
assert y.shape == (8, 8) and quaternion.shape == (1, 4)
assert torch.isfinite(y).all() and torch.isfinite(quaternion).all()
assert np.isfinite(y.sum().item())
print(
    "CLOSD_R3_GPU_WITNESS",
    tuple(y.shape),
    tuple(quaternion.shape),
    torch.cuda.get_device_name(0),
    round(float(y.sum().item()), 6),
    round(float(quaternion.sum().item()), 6),
)
