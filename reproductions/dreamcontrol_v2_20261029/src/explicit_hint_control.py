"""Add an explicit 22-joint mask to an existing OmniControl CMDM instance.

The added 22→latent linear layer is initialized to zero. Existing checkpoint
parameters are left unchanged; the new layer must be trained and saved. This
is an experimental extension, not an author-provided DreamControl-v2 model.
"""
from __future__ import annotations
from types import MethodType
import torch


def _control_with_explicit_mask(self, x, timesteps, y=None, weight=1.0):
    if y is None or 'hint' not in y or 'hint_mask' not in y:
        raise ValueError('explicit hint and hint_mask are required')
    batch, _, _, frames = x.shape
    hint, mask = y['hint'], y['hint_mask']
    if hint.shape != (batch, frames, 66) or not torch.isfinite(hint).all():
        raise ValueError('hint must be finite with shape (B,T,66)')
    if mask.shape == (batch, frames, 22, 1):
        mask = mask.squeeze(-1)
    if mask.shape != (batch, frames, 22) or mask.dtype != torch.bool:
        raise ValueError('hint_mask must be boolean with shape (B,T,22[,1])')
    if mask.device != x.device or hint.device != x.device:
        raise ValueError('hint and mask must be on the same device as motion')
    # Masked-out coordinates must not influence the control network.
    hint = hint.reshape(batch, frames, 22, 3) * mask.unsqueeze(-1)
    hint = hint.reshape(batch, frames, 66)
    seq_mask = mask.any(dim=-1)
    guided_hint = self.input_hint_block(hint.float())
    guided_hint = guided_hint + self.hint_mask_embedding(mask.to(dtype=guided_hint.dtype)).permute(1, 0, 2)
    embedding = self.c_embed_timestep(timesteps)
    if 'text' in self.cond_mode:
        text = self.encode_text(y['text'])
        embedding += self.c_embed_text(self.mask_cond(text, force_mask=y.get('uncond', False)))
    inputs = self.c_input_process(x)
    inputs = inputs + guided_hint * seq_mask.permute(1, 0).unsqueeze(-1)
    inputs = self.c_sequence_pos_encoder(torch.cat((embedding, inputs), dim=0))
    outputs = self.c_seqTransEncoder(inputs)
    return torch.stack([module(outputs[i]) for i, module in enumerate(self.zero_convs)]) * weight


def attach_explicit_hint_mask(model) -> None:
    """Add a fresh mask projection and replace only this instance's control path."""
    if hasattr(model, 'hint_mask_embedding'):
        raise ValueError('explicit mask branch is already attached')
    if not hasattr(model, 'latent_dim') or not hasattr(model, 'cmdm_forward'):
        raise ValueError('expected an OmniControl CMDM instance')
    reference = next(model.parameters())
    projection = torch.nn.Linear(22, model.latent_dim, bias=False, device=reference.device, dtype=reference.dtype)
    torch.nn.init.zeros_(projection.weight)
    model.hint_mask_embedding = projection
    model.cmdm_forward = MethodType(_control_with_explicit_mask, model)
