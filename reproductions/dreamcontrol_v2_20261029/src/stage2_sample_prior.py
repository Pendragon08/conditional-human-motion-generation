"""Sample an immutable trained robot checkpoint; check both conditions and MSE guide."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stage2_cmdm29 import build_model, restore_state, sha
from stage2_robot_prior import setup, preflight, diffusion_for, text_cache, batch_from, sampling_probe, write_json, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'sampling output already exists')
    cfg = json.loads(args.config.read_text())
    cfg['gpu_memory_fraction'] = .05
    _, device = setup(cfg)
    data, stats, provenance = preflight(cfg)
    ckpt_hash = sha(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    require(saved['version'] == 2 and saved['provenance'] == provenance and saved['cursor']['step'] > 0, 'trained robot checkpoint provenance')
    model, model_receipt = build_model(cfg)
    restore_state(model, saved['model'])
    model.to(device)
    embeddings = text_cache(model, data, device)
    diffusion = diffusion_for(stats)
    from robot_spatial_guidance import guide
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    items = [next(r for r in data if r['row']['split'] == 'val' and r['row']['dataset_source'] == source and len(r['motion']) >= 60)
             for source in ('OmniRetarget/OMOMO', 'DexWeave/GRAB')]
    motion, cond = batch_from(items, stats, embeddings, device, training=False, generator=torch.Generator().manual_seed(42))
    model.eval()
    t = torch.full((len(motion),), 200, device=device, dtype=torch.long)
    with torch.no_grad():
        estimate = model(motion, t, **cond)
        null = {**cond['y'], 'uncond': True}
        no_text = model(motion, t, y=null)
        zero_space = {**cond['y'], 'hint': torch.zeros_like(cond['y']['hint']), 'hint_mask': torch.zeros_like(cond['y']['hint_mask'])}
        no_space = model(motion, t, y=zero_space)
    text_difference = float(((estimate - no_text) * cond['y']['mask']).abs().max())
    space_difference = float(((estimate - no_space) * cond['y']['mask']).abs().max())
    require(text_difference > 1e-8 and space_difference > 1e-8, 'text or learned realism condition has no effect')
    def spatial_error(x):
        f = x[:, :, 0].permute(0, 2, 1) * stats['feature_std'].to(device) + stats['feature_mean'].to(device)
        p = recover_from_ric(f, 22)
        h = cond['y']['hint'].reshape(len(x), x.shape[-1], 22, 3)
        goal = h * stats['spatial_std'].to(device).reshape(22, 3) + stats['spatial_mean'].to(device).reshape(22, 3)
        mask = cond['y']['hint_mask'] & cond['y']['mask'][:, 0, 0, :, None]
        return float(((p - goal).square() * mask[..., None]).sum() / (mask.sum() * 3))
    before = spatial_error(estimate)
    guided = guide(diffusion, estimate, t, cond['y']['hint'], cond['y']['hint_mask'],
                   step_size=1., steps=1, valid_frames=cond['y']['mask'][:, 0, 0])
    after = spatial_error(guided)
    require(after <= before + 1e-7, 'local MSE spatial-gradient step increased target error')
    args.output_dir.mkdir(exist_ok=False)
    results = []
    for index, item in enumerate(items):
        sample = sampling_probe(model, diffusion, item, stats, embeddings, device, frames=60, seed=cfg['seed'] + index)
        metadata = {**sample['metadata'], 'checkpoint_sha256': ckpt_hash, 'trained_step': saved['cursor']['step'],
                    'dataset_source': item['row']['dataset_source'], 'validation_reference_id': item['row']['sample_id']}
        path = args.output_dir / ('sample_%d.npz' % index)
        with path.open('xb') as f:
            np.savez_compressed(f, features=sample['features'], canonical_points=sample['canonical_points'],
                                target_points=sample['target_points'], target_mask=sample['target_mask'],
                                metadata_utf8=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8))
        results.append({'sample': path.name, 'sha256': sha(path), **metadata})
    require(sha(args.checkpoint) == ckpt_hash, 'checkpoint changed while sampling')
    report = {'checkpoint_sha256': ckpt_hash, 'trained_step': saved['cursor']['step'],
              'text_condition_max_output_difference': text_difference,
              'realism_spatial_condition_max_output_difference': space_difference,
              'local_spatial_mse_before': before, 'local_spatial_mse_after': after,
              'samples': results, 'scope': 'trained-checkpoint inference and conditioning witness, not task-success/physics acceptance'}
    write_json(args.output_dir / 'report.json', report, exclusive=True)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
