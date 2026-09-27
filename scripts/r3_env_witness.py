"""Minimal deterministic GPU/dependency witness for the R3 DreamControl overlay."""

import torch
import joblib
import easydict
import loguru
import pytorch_kinematics
import jaxlie

torch.manual_seed(123)
x = torch.randn(8, 8, device="cuda:0")
y = x @ x
torch.cuda.synchronize(0)
print(
    "DREAMCONTROL_R3_GPU_WITNESS",
    tuple(y.shape),
    torch.cuda.get_device_name(0),
    float(y.abs().sum()),
    flush=True,
)
