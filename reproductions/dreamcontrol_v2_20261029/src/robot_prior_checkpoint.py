"""Load an explicitly identified experimental robot-training checkpoint.

This verifies bytes, training provenance, completed steps and model keys before
loading; it does not judge dataset quality or establish v2 paper fidelity.
Only load trusted checkpoints.  A human-only pretrained state dict is rejected.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import torch


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_robot_prior_checkpoint(model, checkpoint_path, *, checkpoint_sha256,
                                stats_path, dataset_manifest_sha256):
    """Load a trained project's checkpoint only with matching pinned inputs.

    The model architecture and explicit hint branch must already be attached.
    The caller independently decides whether provenance is eligible for use.
    This function does not restore optimizer state or random generators.
    """
    for value in (checkpoint_sha256,dataset_manifest_sha256):
        if not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None:
            raise ValueError('expected lowercase SHA-256 pins')
    checkpoint=Path(checkpoint_path).resolve(strict=True);stats=Path(stats_path).resolve(strict=True)
    if not checkpoint.is_file() or not stats.is_file():raise ValueError('checkpoint and statistics files required')
    if _sha(checkpoint)!=checkpoint_sha256:raise ValueError('checkpoint hash mismatch')
    stats_sha=_sha(stats)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    required={'version','model','optimizer','rng','cursor','config','provenance'}
    if not isinstance(payload,dict) or not required.issubset(payload) or payload.get('version')!=1 or not isinstance(payload.get('model'),dict) or not isinstance(payload.get('config'),dict):
        raise ValueError('expected project robot-training checkpoint, not a base model')
    provenance=payload.get('provenance');cursor=payload.get('cursor')
    if (not isinstance(provenance,dict) or provenance.get('dataset_manifest_sha256')!=dataset_manifest_sha256 or
        provenance.get('stats_sha256')!=stats_sha or not isinstance(provenance.get('model_init_sha256'),str) or
        re.fullmatch(r'[0-9a-f]{64}',provenance['model_init_sha256']) is None):
        raise ValueError('training manifest or statistics provenance mismatch')
    if (not isinstance(cursor,dict) or set(cursor)!={'step','epoch','next_batch'} or
        any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in cursor.values()) or cursor['step']<1):
        raise ValueError('checkpoint must record at least one completed training step')
    if not hasattr(model,'hint_mask_embedding'):raise ValueError('explicit hint-mask branch must be attached first')
    expected=model.state_dict();stored=payload['model']
    if set(expected)!=set(stored):raise ValueError('checkpoint model keys differ from sampling model')
    for name,tensor in stored.items():
        target=expected[name]
        if not isinstance(tensor,torch.Tensor) or tensor.shape!=target.shape or tensor.dtype!=target.dtype:
            raise ValueError('checkpoint tensor shape or dtype mismatch: '+name)
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not torch.isfinite(tensor).all():
            raise ValueError('checkpoint tensor contains nonfinite values: '+name)
    if _sha(checkpoint)!=checkpoint_sha256 or _sha(stats)!=stats_sha:
        raise RuntimeError('checkpoint or statistics changed while loading')
    model.load_state_dict(stored,strict=True)
    return {'checkpoint_sha256':checkpoint_sha256,'stats_sha256':stats_sha,
            'dataset_manifest_sha256':dataset_manifest_sha256,'step':cursor['step'],
            'configuration':payload.get('config'),'training_or_physical_acceptance':False}
