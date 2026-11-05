"""Generate a provenance-bound experimental robot-prior sample bundle.

The caller constructs the original CMDM, attaches the explicit mask branch,
chooses a safe device, and validates that its robot checkpoint is eligible.
This wrapper refuses pretrained-only weights and never overwrites an output.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import numpy as np
from robot_diffusion_stats import configure_robot_stats
from robot_prior_checkpoint import load_robot_prior_checkpoint
from robot_prior_sampling import sample_robot_prior


def _sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def generate_robot_prior(model,diffusion,*,checkpoint_path,checkpoint_sha256,manifest_path,stats_path,
                         output_path,texts,targets,target_mask,cfg_scale,spatial_step_size,spatial_steps,
                         seed,min_free_bytes):
    """Write one atomic NPZ with generated features, points and provenance.

    Input checkpoint must be a matching, nonzero-step project checkpoint. This
    does not independently accept data quality or authorize real deployment.
    min_free_bytes is a caller-specified reserve on the output filesystem.
    """
    if isinstance(min_free_bytes,bool) or not isinstance(min_free_bytes,int) or min_free_bytes<0:
        raise ValueError('explicit nonnegative disk reserve required')
    out=Path(output_path)
    if out.suffix!='.npz' or not out.parent.is_dir():raise ValueError('output must be an NPZ in an existing directory')
    if out.exists():raise FileExistsError('output already exists')
    if shutil.disk_usage(out.parent).free<min_free_bytes:raise RuntimeError('insufficient output disk reserve')
    manifest=Path(manifest_path).resolve(strict=True);stats=Path(stats_path).resolve(strict=True)
    if not manifest.is_file() or not stats.is_file():raise ValueError('manifest and statistics files required')
    manifest_sha,stats_sha=_sha(manifest),_sha(stats)
    with np.load(stats,allow_pickle=False) as f:
        arrays=[f[k].copy() for k in ('feature_mean','feature_std','spatial_mean','spatial_std')]
    # Validate stats before mutating either the model or supplied diffusion.
    configure_robot_stats(SimpleNamespace(),*arrays)
    loaded=load_robot_prior_checkpoint(model,checkpoint_path,checkpoint_sha256=checkpoint_sha256,
                                       stats_path=stats,dataset_manifest_sha256=manifest_sha)
    configure_robot_stats(diffusion,*arrays)
    model.eval()
    sample=sample_robot_prior(model,diffusion,texts=texts,targets=targets,target_mask=target_mask,
                              cfg_scale=cfg_scale,spatial_step_size=spatial_step_size,
                              spatial_steps=spatial_steps,seed=seed)
    if _sha(manifest)!=manifest_sha or _sha(stats)!=stats_sha or _sha(Path(checkpoint_path))!=checkpoint_sha256:
        raise RuntimeError('generation inputs changed during sampling')
    metadata={'checkpoint':loaded,'sampling':sample['metadata'],'texts':texts,
              'manifest_sha256':manifest_sha,'stats_sha256':stats_sha,
              'training_or_physical_acceptance':False}
    metadata_bytes=json.dumps(metadata,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf8')
    temp=None
    try:
        if shutil.disk_usage(out.parent).free<min_free_bytes:raise RuntimeError('output disk reserve fell during sampling')
        with tempfile.NamedTemporaryFile(prefix='.robot-prior-',suffix='.tmp',dir=out.parent,delete=False) as stream:
            temp=Path(stream.name)
            np.savez_compressed(stream,normalized_features=sample['normalized_features'].cpu().numpy(),
                                canonical_points=sample['canonical_points'].cpu().numpy(),
                                targets=targets.detach().cpu().numpy(),target_mask=target_mask.detach().cpu().numpy(),
                                metadata_utf8=np.frombuffer(metadata_bytes,dtype=np.uint8))
            stream.flush();os.fsync(stream.fileno())
        if shutil.disk_usage(out.parent).free<min_free_bytes:raise RuntimeError('output disk reserve fell while saving')
        os.link(temp,out)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)
    return {'output_path':str(out),'output_sha256':_sha(out),'metadata':metadata,
            'training_or_physical_acceptance':False}
