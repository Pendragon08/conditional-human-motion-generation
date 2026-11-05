"""Guarded single-process robot-prior training entry point.

Currently feature audit cannot verify physical data quality, so --run fails
closed until a genuine physical/source-quality validator is integrated. The
implementation uses an OmniControl human checkpoint and experimental explicit
mask/MSE guidance; this must be recorded as a reproduction approximation.
Run from the pinned DreamControl TrajGen checkout with this directory and CLIP
on PYTHONPATH. Do not use this entry for sim-to-real deployment.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import numpy as np
import torch
from audit_robot_features import audit, sha256_file


def state_preflight(root, records, cfg):
    """Run optional recomputed G1 state checks; missing evidence fails closed."""
    keys = ('state_root', 'state_manifest', 'state_urdf', 'state_urdf_sha256')
    present = [key in cfg for key in keys]
    if not any(present):
        return {'state_technical_checks_passed': False, 'status': 'missing_state_evidence', 'training_ready': False}
    if not all(present):
        raise ValueError('state evidence requires state_root, state_manifest, state_urdf and state_urdf_sha256')
    state_manifest = Path(cfg['state_manifest']).resolve(strict=True)
    state_urdf = Path(cfg['state_urdf']).resolve(strict=True)
    state_hash, urdf_hash = sha256_file(state_manifest), sha256_file(state_urdf)
    if urdf_hash != cfg['state_urdf_sha256']:
        raise ValueError('state URDF does not match pinned hash')
    evidence = json.loads(state_manifest.read_text(encoding='utf8'))
    if not isinstance(evidence.get('records'), list):
        raise ValueError('state evidence records required')
    index = {}
    for row in evidence['records']:
        sample = row.get('sample_id')
        if not isinstance(sample, str) or not sample or sample in index:
            raise ValueError('duplicate or missing state evidence ID')
        index[sample] = row
    if len(index) != len(records) or set(index) != {row['sample_id'] for row in records}:
        raise ValueError('state evidence must cover exactly the dataset manifest')
    combined = []
    for row in records:
        state = index[row['sample_id']]
        if state.get('feature_sha256') != row['feature_sha256']:
            raise ValueError('state evidence refers to different features')
        combined.append({**row, 'state_path': state.get('state_path'), 'state_sha256': state.get('state_sha256'),
                         'foot_velocity_squared_threshold': state.get('foot_velocity_squared_threshold'),
                         'encoding_variant': state.get('encoding_variant', 'legacy')})
    from audit_robot_states import audit_robot_states
    result = audit_robot_states(combined, feature_root=root, state_root=cfg['state_root'], urdf_path=state_urdf,
                                tolerance_rad=1e-6, fk_decode_tolerance_m=1e-4, reencoding_tolerance=1e-5)
    if sha256_file(state_manifest) != state_hash or sha256_file(state_urdf) != urdf_hash:
        raise RuntimeError('state evidence or URDF changed during preflight')
    result['state_manifest_sha256'] = state_hash
    return result


def resource_preflight(device, run_dir, *, checkpoint_size_estimate, max_steps, checkpoint_every,
                       allow_shared_gpu=False, gpu_memory_fraction=None):
    if not isinstance(checkpoint_size_estimate,int) or checkpoint_size_estimate < 1024**3:
        raise ValueError('checkpoint_size_estimate must be at least 1 GiB and supported by measurement')
    if not isinstance(max_steps,int) or max_steps < 1 or not isinstance(checkpoint_every,int) or checkpoint_every < 1:
        raise ValueError('invalid step or checkpoint interval')
    # Checkpoints are never deleted automatically, including the final one.
    planned = math.ceil(max_steps/checkpoint_every)
    # Preserve the project's 8 GiB stop line after all planned checkpoint
    # allocations, including the two extra checkpoint-size safety allowances.
    needed = (planned+2)*checkpoint_size_estimate + 8*1024**3
    if shutil.disk_usage(run_dir).free < needed:
        raise RuntimeError(f'insufficient free disk for estimated checkpoint budget: need {needed} bytes')
    if device.type != 'cuda' or device.index is None:
        raise ValueError('formal training requires an explicitly indexed CUDA device')
    if not isinstance(allow_shared_gpu, bool):
        raise ValueError('allow_shared_gpu must be boolean')
    if allow_shared_gpu:
        if ('CUDA_VISIBLE_DEVICES' in os.environ):
            raise ValueError('shared GPU preflight requires CUDA_VISIBLE_DEVICES unset so physical and CUDA indices agree')
        if (isinstance(gpu_memory_fraction, bool) or not isinstance(gpu_memory_fraction, (int, float)) or
            not math.isfinite(gpu_memory_fraction) or not 0 < gpu_memory_fraction <= .10):
            raise ValueError('shared GPU requires an explicit memory fraction in (0, 0.10]')
    elif gpu_memory_fraction is not None:
        raise ValueError('gpu_memory_fraction requires explicit allow_shared_gpu')
    gpu_lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free',
                                       '--format=csv,noheader,nounits'],text=True).splitlines()
    gpus={int(index.strip()):(uuid.strip(),int(free.strip())) for index,uuid,free in
          (line.split(',') for line in gpu_lines if line.strip())}
    if device.index not in gpus: raise ValueError('requested GPU does not exist')
    uuid, free_mb=gpus[device.index]
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                       '--format=csv,noheader,nounits'],text=True).splitlines()
    if any(line.split(',')[0].strip()==uuid for line in processes if ',' in line) and not allow_shared_gpu:
        raise RuntimeError('requested GPU has an existing compute task')
    if free_mb < 10240: raise RuntimeError('requested GPU has less than 10 GiB free')
    if allow_shared_gpu:
        totals=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total',
                                        '--format=csv,noheader,nounits'],text=True).splitlines()
        totals={int(index.strip()):int(total.strip()) for index,total in
                (line.split(',') for line in totals if line.strip())}
        if device.index not in totals or free_mb < totals[device.index] * float(gpu_memory_fraction) + 2048:
            raise RuntimeError('insufficient free GPU memory for the requested fraction and 2 GiB headroom')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',action='store_true',help='request training after all acceptance checks')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    required=('dataset_root','manifest','stats','initial_checkpoint','run_dir','min_std','batch_size','seed',
              'max_steps','checkpoint_every','guidance_steps','guidance_scale','learning_rate','weight_decay',
              'device','checkpoint_size_estimate')
    missing=[key for key in required if key not in cfg]
    if missing: raise ValueError(f'missing configuration fields: {missing}')
    root=Path(cfg['dataset_root']).resolve(strict=True)
    manifest_path=Path(cfg['manifest']).resolve(strict=True)
    stats_path=Path(cfg['stats']).resolve(strict=True)
    init_path=Path(cfg['initial_checkpoint']).resolve(strict=True)
    run_dir=Path(cfg['run_dir']).resolve(strict=True)
    manifest_hash,stats_hash=sha256_file(manifest_path),sha256_file(stats_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    with np.load(stats_path,allow_pickle=False) as f:
        stats={key:f[key].copy() for key in ('feature_mean','feature_std','spatial_mean','spatial_std')}
    report=audit(root,manifest['records'],stats,min_std=cfg['min_std'])
    states=state_preflight(root,manifest['records'],cfg)
    allowed_origins={'source_text','manual_visual','machine_geometry_v1','machine_geometry_v2'}
    if any(r.get('caption_origin','source_text') not in allowed_origins for r in manifest['records']):
        raise ValueError('unsupported caption origin')
    source_captions=[r for r in manifest['records'] if r.get('caption_origin','source_text')=='source_text']
    manual_captions=[r for r in manifest['records'] if r.get('caption_origin','source_text')=='manual_visual']
    geometry_captions=[r for r in manifest['records'] if r.get('caption_origin','source_text') in ('machine_geometry_v1','machine_geometry_v2')]
    from audit_caption_receipts import audit_caption_receipts
    captions=audit_caption_receipts(source_captions,evidence_root=cfg.get('caption_evidence_root'),
                                    evidence_manifest=cfg.get('caption_evidence_manifest')) if source_captions else {'status':'no_source_captions','training_ready':False}
    from audit_manual_caption_binding import audit_manual_captions
    manual_cfg=cfg.get('manual_caption_binding')
    if manual_cfg is not None and not isinstance(manual_cfg,dict):raise ValueError('manual_caption_binding must be an object')
    manual_binding=audit_manual_captions(manual_captions,**manual_cfg) if manual_cfg is not None else audit_manual_captions(manual_captions)
    from audit_geometry_caption_binding import audit_geometry_captions
    geometry_cfg=cfg.get('geometry_caption_binding')
    if geometry_cfg is not None and not isinstance(geometry_cfg,dict):raise ValueError('geometry_caption_binding must be an object')
    geometry_binding=audit_geometry_captions(geometry_captions,**geometry_cfg) if geometry_cfg is not None else audit_geometry_captions(geometry_captions)
    from audit_omniretarget_release_binding import audit_release_binding
    release_cfg=cfg.get('omniretarget_release_binding')
    if release_cfg is not None and not isinstance(release_cfg,dict):raise ValueError('omniretarget_release_binding must be an object')
    release_binding=audit_release_binding(manifest['records'],**release_cfg) if release_cfg is not None else audit_release_binding(manifest['records'])
    from audit_caption_archive_binding import audit_caption_archive
    archive_cfg=cfg.get('caption_archive_binding')
    if archive_cfg is not None and not isinstance(archive_cfg,dict):
        raise ValueError('caption_archive_binding must be an object')
    archive_binding=audit_caption_archive(source_captions, **archive_cfg) if archive_cfg is not None else audit_caption_archive(source_captions)
    from audit_ssm_source_evidence import audit_ssm_source_bundle
    source_evidence=audit_ssm_source_bundle(manifest['records'],alignment_path=cfg.get('ssm_alignment_path'),
                                            evidence_manifest=cfg.get('ssm_source_evidence_manifest'))
    from audit_vertical_support_preflight import audit_vertical_support
    vertical_cfg=cfg.get('vertical_support_diagnostic')
    if vertical_cfg is not None and not isinstance(vertical_cfg,dict):
        raise ValueError('vertical_support_diagnostic must be an object')
    vertical = audit_vertical_support(manifest['records'], **vertical_cfg) if vertical_cfg is not None else audit_vertical_support(manifest['records'])
    length_cfg=cfg.get('legacy_length_diagnostic')
    if length_cfg is not None and not isinstance(length_cfg,dict):raise ValueError('legacy_length_diagnostic must be an object')
    if length_cfg is not None:
        from audit_legacy_length_and_batches import audit_lengths
        length_diagnostic=audit_lengths(manifest['records'],root=root,batch_size=cfg['batch_size'],seed=cfg['seed'],**length_cfg)
    else:
        length_diagnostic={'status':'not_requested','training_ready':False}
    if sha256_file(manifest_path)!=manifest_hash or sha256_file(stats_path)!=stats_hash:
        raise RuntimeError('manifest or statistics changed during preflight')
    print(json.dumps({'feature_audit':report,'state_audit':states,'caption_evidence_audit':captions,'manual_caption_binding':manual_binding,'geometry_caption_binding':geometry_binding,'omniretarget_release_binding':release_binding,
                      'caption_archive_binding':archive_binding,
                      'ssm_source_evidence_audit':source_evidence,
                      'vertical_support_diagnostic':vertical,
                      'legacy_length_and_batch_diagnostic':length_diagnostic,
                      'manifest_sha256':manifest_hash,'stats_sha256':stats_hash},ensure_ascii=False))
    if not args.run: return
    if not report['training_ready']:
        raise RuntimeError('training blocked: source and applicable quality acceptance not yet implemented')
    if not states['state_technical_checks_passed']:
        raise RuntimeError('training blocked: state evidence missing or joint-limit checks failed')
    device=torch.device(cfg['device'])
    allow_shared_gpu=cfg.get('allow_shared_gpu',False)
    gpu_memory_fraction=cfg.get('gpu_memory_fraction')
    resource_preflight(device,run_dir,checkpoint_size_estimate=cfg['checkpoint_size_estimate'],
                       max_steps=cfg['max_steps'],checkpoint_every=cfg['checkpoint_every'],
                       allow_shared_gpu=allow_shared_gpu,gpu_memory_fraction=gpu_memory_fraction)
    if allow_shared_gpu:
        # Caps PyTorch's caching allocator, not every CUDA allocation by other libraries.
        torch.cuda.set_per_process_memory_fraction(float(gpu_memory_fraction),device)
    from diffusion import gaussian_diffusion as gd
    from model.cmdm import CMDM
    from utils.model_util import get_model_args
    from types import SimpleNamespace
    from explicit_hint_control import attach_explicit_hint_mask
    from robot_diffusion_stats import configure_robot_stats
    from robot_motion_dataset import RobotMotionDataset
    from robot_training_loop import run_training
    settings=SimpleNamespace(dataset='humanml',latent_dim=512,layers=8,cond_mode='both_text_spatial',
                             cond_mask_prob=.1,arch='trans_enc',emb_trans_dec=False)
    torch.manual_seed(cfg['seed'])
    model=CMDM(**get_model_args(settings,None))
    weights=torch.load(init_path,map_location='cpu',weights_only=True)
    missing,unexpected=model.load_state_dict(weights,strict=False)
    if unexpected or any(not key.startswith('clip_model.') for key in missing):
        raise ValueError('initial checkpoint does not match CMDM architecture')
    attach_explicit_hint_mask(model)
    model.to(device)
    diffusion=gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('cosine',1000,1.),
                                   model_mean_type=gd.ModelMeanType.START_X,model_var_type=gd.ModelVarType.FIXED_SMALL,
                                   loss_type=gd.LossType.MSE,dataset='humanml')
    configure_robot_stats(diffusion,stats['feature_mean'],stats['feature_std'],stats['spatial_mean'],stats['spatial_std'])
    dataset=RobotMotionDataset(root=root,records=manifest['records'],split='train',
                               feature_mean=stats['feature_mean'],feature_std=stats['feature_std'],
                               spatial_mean=stats['spatial_mean'],spatial_std=stats['spatial_std'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    provenance={'dataset_manifest_sha256':manifest_hash,'stats_sha256':stats_hash,'model_init_sha256':sha256_file(init_path),
                'state_manifest_sha256':states['state_manifest_sha256'],'state_urdf_sha256':states['urdf_sha256']}
    # max_steps can be extended on resume; remaining configuration is immutable.
    recorded={k:v for k,v in cfg.items() if k not in ('max_steps',)}
    result=run_training(model,diffusion,optimizer,dataset,run_dir=run_dir,provenance=provenance,config=recorded,
                        max_steps=cfg['max_steps'],checkpoint_every=cfg['checkpoint_every'],batch_size=cfg['batch_size'],
                        seed=cfg['seed'],guidance_steps=cfg['guidance_steps'],guidance_scale=cfg['guidance_scale'],
                        device=device,resume_from=args.resume,
                        accumulate_microbatches=cfg.get('accumulate_microbatches',1),
                        on_step=lambda metrics,cursor:print(json.dumps({'cursor':cursor,'metrics':metrics}),flush=True))
    print(json.dumps(result))


if __name__=='__main__': main()
