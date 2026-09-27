"""Stage the official ViT-B/32 text encoder used by CLoSD and test a CPU pass."""

import torch
import clip


model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
model.eval()
with torch.no_grad():
    tokens = clip.tokenize(["a person walks forward"])
    output = model.encode_text(tokens)
assert output.shape == (1, 512) and torch.isfinite(output).all()
print("CLOSD_R3_CLIP_WITNESS", tuple(output.shape), round(float(output.norm().item()), 6))
