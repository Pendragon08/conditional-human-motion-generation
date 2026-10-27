"""Lossless active-expert FP32 inference artifact; never a resume checkpoint."""
import re
import torch

KIND='phygile_active_policy_fp32_inference_only_v1'

def omitted_key(name):
    return name.startswith('critic.') or bool(re.match(r'experts\.(?:[1-9]\d*)\.',name))

def load_active(model,payload,contract):
    if payload.get('artifact_kind')!=KIND or payload.get('active_experts')!=1 or contract['active_experts']!=1:
        raise ValueError('Compact inference artifact only supports exactly one active expert')
    expected={k for k in model.state_dict() if omitted_key(k)}
    if set(payload['omitted_state_keys'])!=expected:raise ValueError('Omitted branch schema differs')
    if set(payload['model_state_dict'])!={k for k in model.state_dict() if k not in expected}:
        raise ValueError('Active inference tensor set differs')
    incompatible=model.load_state_dict(payload['model_state_dict'],strict=False)
    if set(incompatible.missing_keys)!=expected or incompatible.unexpected_keys:
        raise ValueError('Unexpected incomplete state')
    def forbidden(*_):raise RuntimeError('Compact inference entered an omitted branch')
    model.critic.register_forward_pre_hook(forbidden)
    for expert in list(model.experts)[1:]:expert.register_forward_pre_hook(forbidden)
    return True
