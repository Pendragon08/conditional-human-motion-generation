"""Isolated, resumable 29DoF-dataset CMDM baseline: preflight/smoke/train/sample.

Training conditions are published sequence-level captions and sampled 22-point
spatial targets. No source-trajectory physics/semantic screen is introduced.
All unspecified v2 hyperparameters are explicit baseline configuration choices.
"""
import argparse
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from stage2_cmdm29 import build_model, compact_state, restore_state, sha


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, data, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x', encoding='utf8') as f:
            json.dump(data, f, indent=2, allow_nan=False)
        return
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w', encoding='utf8') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup(cfg):
    root = Path(cfg['project_root'])
    traj = Path(cfg['trajgen'])
    sys.path[:0] = [str(root / 'clip_reuse'), str(traj), str(root / 'training_disk_guard_20261101')]
    os.chdir(traj)
    torch.set_num_threads(cfg['cpu_threads'])
    device = torch.device(cfg['device'])
    require('CUDA_VISIBLE_DEVICES' not in os.environ, 'physical CUDA indices must be preserved')
    require(device.type == 'cuda' and device.index is not None, 'explicit GPU index required')
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.total',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    inventory = {int(a): (int(b), int(c)) for a, b, c in (line.split(',') for line in lines)}
    free, total = inventory[device.index]
    fraction = cfg['gpu_memory_fraction']
    require(0 < fraction <= .10 and free > total * fraction + 2048, 'shared GPU memory headroom')
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.cuda.set_device(device)
    seed_all(cfg['seed'])
    return root, device


def preflight(cfg):
    root = Path(cfg['dataset_root'])
    manifest = Path(cfg['manifest'])
    stats_path = Path(cfg['stats'])
    require(cfg['data_policy'] == 'published_caption_weak_baseline_no_source_physical_filter', 'unsupported data policy')
    require(sha(manifest) == cfg['manifest_sha256'] and sha(stats_path) == cfg['stats_sha256'], 'dataset inputs changed')
    receipt_path = Path(cfg['completion_receipt'])
    receipt = json.loads(receipt_path.read_text())
    require(receipt['paper_scope_dataset_construction_complete'] and receipt['trajectory_records'] == 3161,
            'paper-scope data construction incomplete')
    require(receipt['manifest_sha256'] == sha(manifest) and receipt['stats_sha256'] == sha(stats_path), 'receipt binding')
    require(receipt['joint_count'] == 29 and receipt['nominal_fps'] == 20, '29DoF/20Hz receipt contract')
    lineage = json.loads(Path(cfg['lineage_report']).read_text())
    require(lineage['passed_source_associations'] == 3161 and not lineage['errors'] and
            lineage['input_sha256']['mixed_manifest'] == sha(manifest), 'source/text lineage report')
    rows = json.loads(manifest.read_text())['records']
    require(len(rows) == 3161 and len({r['sample_id'] for r in rows}) == 3161, 'manifest coverage')
    require(Counter(r['split'] for r in rows) == {'train': 2091, 'val': 484, 'test': 586}, 'fixed dataset splits')
    require(Counter(r['dataset_source'] for r in rows) == {'DexWeave/GRAB': 1335, 'OmniRetarget/OMOMO': 1826}, 'source coverage')
    stats = {}
    with np.load(stats_path, allow_pickle=False) as f:
        for key, size in [('feature_mean', 263), ('feature_std', 263), ('spatial_mean', 66), ('spatial_std', 66)]:
            stats[key] = torch.tensor(np.asarray(f[key]).reshape(-1), dtype=torch.float32)
            require(stats[key].numel() == size and torch.isfinite(stats[key]).all(), 'normalization statistics')
            if key.endswith('std'):
                require((stats[key] > 0).all(), 'positive standard deviations required')
    if sha(Path(cfg['clip_checkpoint'])) != cfg['clip_sha256']:
        raise ValueError('frozen CLIP weights changed')
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    cache = []
    for row in rows:
        path = (root / row['feature_path']).resolve(strict=True)
        require(path.is_relative_to(root) and sha(path) == row['feature_sha256'], 'current feature artifact hash')
        with np.load(path, allow_pickle=False) as z:
            raw = torch.tensor(np.asarray(z['features']), dtype=torch.float32)
            require(len(z['joint_names']) == 29, '29DoF artifact')
            times = np.asarray(z['sample_times'])
        require(raw.ndim == 2 and raw.shape[1] == 263 and 2 <= len(raw) <= 199 and torch.isfinite(raw).all(), 'feature layout')
        require(row['fps'] == 20 and len(times) == len(raw) + 1 and np.allclose(times, np.arange(len(times)) / 20., atol=1e-10), '20Hz artifact timeline')
        with torch.no_grad():
            points = recover_from_ric(raw, 22)
        captions = row.get('candidate_captions', [row['caption']])
        require(captions and row['caption'] in captions and all(isinstance(s, str) and s.strip() for s in captions), 'source text')
        cache.append({'row': row, 'motion': (raw - stats['feature_mean']) / stats['feature_std'],
                      'points': points, 'captions': captions})
    result = {'dataset_manifest_sha256': sha(manifest), 'stats_sha256': sha(stats_path),
              'completion_receipt_sha256': sha(receipt_path), 'lineage_report_sha256': sha(Path(cfg['lineage_report'])),
              'model_init_sha256': cfg['initial_checkpoint_sha256'], 'clip_sha256': cfg['clip_sha256'],
              'rows': len(cache), 'splits': dict(Counter(r['split'] for r in rows)),
              'data_policy': cfg['data_policy'], 'input_source_physical_filter': False,
              'independent_caption_semantics_certified': False, 'scope': 'published-source weak-label robot diffusion baseline'}
    return cache, stats, result


def diffusion_for(stats):
    from diffusion import gaussian_diffusion as gd
    from robot_diffusion_stats import configure_robot_stats
    diffusion = gd.GaussianDiffusion(betas=gd.get_named_beta_schedule('cosine', 1000, 1.),
                                   model_mean_type=gd.ModelMeanType.START_X,
                                   model_var_type=gd.ModelVarType.FIXED_SMALL,
                                   loss_type=gd.LossType.MSE, dataset='humanml')
    configure_robot_stats(diffusion, *(stats[k].numpy() for k in ['feature_mean', 'feature_std', 'spatial_mean', 'spatial_std']))
    return diffusion


def text_cache(model, data, device):
    texts = sorted({s for r in data for s in r['captions']})
    embeddings = {}
    with torch.no_grad():
        for start in range(0, len(texts), 32):
            values = model.encode_text(texts[start:start + 32]).detach()
            embeddings.update(zip(texts[start:start + 32], values.unbind(0)))
    return embeddings


def batch_from(data, stats, embeddings, device, *, training=True, generator=None):
    batch, n = len(data), max(len(r['motion']) for r in data)
    motion = torch.zeros((batch, 263, 1, n), device=device)
    valid = torch.zeros((batch, n), device=device, dtype=torch.bool)
    hint = torch.zeros((batch, n, 22, 3), device=device)
    hm = torch.zeros((batch, n, 22), device=device, dtype=torch.bool)
    captions, lengths = [], []
    for i, row in enumerate(data):
        length = len(row['motion'])
        lengths.append(length)
        valid[i, :length] = True
        motion[i, :, 0, :length] = row['motion'].T.to(device)
        # Support any of the 22 point targets, both sparse and dense in time.
        joint_count = int(torch.randint(1, 3, (), generator=generator))
        joints = torch.randperm(22, generator=generator)[:joint_count]
        count = int(torch.randint(1, length + 1, (), generator=generator))
        frames = torch.randperm(length, generator=generator)[:count]
        for joint in joints.tolist():
            hm[i, frames.to(device), joint] = True
        if training and random.random() < .1:
            hm[i] = False
        points = row['points'].to(device)
        normalized = ((points.reshape(length, 66) - stats['spatial_mean'].to(device)) /
                      stats['spatial_std'].to(device)).reshape(length, 22, 3)
        hint[i, :length] = normalized * hm[i, :length, :, None]
        captions.append(random.choice(row['captions']) if training else row['captions'][0])
    y = {'mask': valid[:, None, None], 'hint_mask': hm, 'hint': hint.reshape(batch, n, 66),
         'lengths': torch.tensor(lengths, device=device), 'text': captions,
         'text_embedding': torch.stack([embeddings[s] for s in captions])}
    return motion, {'y': y}


def loss_for(model, diffusion, motion, condition, timesteps=None, noise=None):
    if timesteps is None:
        timesteps = torch.randint(diffusion.num_timesteps, (len(motion),), device=motion.device)
    if noise is None:
        noise = torch.randn_like(motion)
    prediction = model(diffusion.q_sample(motion, timesteps, noise=noise), timesteps, **condition)
    mask = condition['y']['mask']
    per_sample = ((prediction - motion).square() * mask).sum((1, 2, 3)) / (mask.sum((1, 2, 3)) * 263)
    require(torch.isfinite(per_sample).all(), 'nonfinite denoising loss')
    return per_sample.mean()


def evaluate(model, diffusion, data, stats, embeddings, device, batch_size=16):
    model.eval()
    losses = []
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(20261103)
        g = torch.Generator().manual_seed(20261103)
        with torch.no_grad():
            for start in range(0, len(data), batch_size):
                items = data[start:start + batch_size]
                motion, cond = batch_from(items, stats, embeddings, device, training=False, generator=g)
                losses.append((float(loss_for(model, diffusion, motion, cond)), len(items)))
    model.train()
    model.clip_model.eval()
    return sum(x * n for x, n in losses) / sum(n for _, n in losses)


def rng_state(device):
    name, values, pos, has_gauss, cached = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [name, values.tolist(), pos, has_gauss, cached],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)}


def restore_rng(state, device):
    random.setstate(state['python'])
    name, values, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def payload(model, optimizer, cursor, cfg, provenance, device):
    return {'version': 2, 'model': compact_state(model), 'optimizer': optimizer.state_dict(),
            'cursor': cursor, 'config': cfg, 'provenance': provenance, 'rng': rng_state(device)}


def checkpoint(path, data, cfg):
    path = Path(path)
    require(not path.exists(), 'never overwrite checkpoints')
    require(shutil.disk_usage(path.parent).free >= cfg['disk_floor_bytes'] + cfg['checkpoint_size_budget'], 'checkpoint disk reserve')
    temporary = path.with_suffix('.partial')
    require(not temporary.exists(), 'checkpoint partial already exists')
    with temporary.open('xb') as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    require(temporary.stat().st_size <= cfg['checkpoint_size_budget'], 'checkpoint exceeded measured size budget')
    require(shutil.disk_usage(path.parent).free >= cfg['disk_floor_bytes'], 'disk reserve after checkpoint')
    os.link(temporary, path)
    temporary.unlink()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def sampling_probe(model, diffusion, item, stats, embeddings, device, frames=60, seed=123):
    from model.cfg_sampler import ClassifierFreeSampleModel
    from robot_spatial_guidance import guide
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    model.eval()
    n = min(frames, len(item['motion']))
    target = item['points'][:n].to(device)
    mask = torch.zeros((1, n, 22), device=device, dtype=torch.bool)
    mask[0, n // 2, 21] = True
    normalized = ((target.reshape(n, 66) - stats['spatial_mean'].to(device)) / stats['spatial_std'].to(device))
    caption = item['captions'][0]
    y = {'text': [caption], 'text_embedding': embeddings[caption].unsqueeze(0),
         'hint': (normalized.reshape(1, n, 22, 3) * mask.unsqueeze(-1)).reshape(1, n, 66),
         'hint_mask': mask, 'mask': torch.ones((1, 1, 1, n), device=device, dtype=torch.bool),
         'lengths': torch.tensor([n], device=device), 'scale': torch.tensor([2.5], device=device)}
    original_guide = diffusion.guide
    def adapter(self, x, t, model_kwargs=None, **kwargs):
        yy = model_kwargs['y']
        return guide(self, x, t, yy['hint'], yy['hint_mask'], step_size=1., steps=1,
                     valid_frames=yy['mask'][:, 0, 0])
    diffusion.guide = MethodType(adapter, diffusion)
    began = time.monotonic()
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed)
            wrapped = ClassifierFreeSampleModel(model)
            normalized_features = diffusion.p_sample_loop(wrapped, (1, 263, 1, n), clip_denoised=False,
                                                          model_kwargs={'y': y}, device=device, progress=False)
        features = normalized_features[:, :, 0].permute(0, 2, 1) * stats['feature_std'].to(device) + stats['feature_mean'].to(device)
        points = recover_from_ric(features, 22)
        require(torch.isfinite(points).all() and torch.isfinite(features).all(), 'nonfinite sampled motion')
        error = float(torch.linalg.vector_norm(points[0, n // 2, 21] - target[n // 2, 21]))
        return {'features': features.detach().cpu().numpy(), 'canonical_points': points.detach().cpu().numpy(),
                'target_points': target.detach().cpu().numpy(), 'target_mask': mask.cpu().numpy(),
                'metadata': {'caption': caption, 'frames': n, 'fps': 20, 'diffusion_steps': 1000,
                             'cfg_scale': 2.5, 'spatial_gradient_steps': 1, 'spatial_step_size': 1.,
                             'right_wrist_target_frame': n // 2, 'right_wrist_error_m': error,
                             'elapsed_s': time.monotonic() - began, 'seed': seed,
                             'space': 'canonical 22-point coordinates, not scene-calibrated world coordinates',
                             'joint_dof_after_postoptimization': 29, 'physical_acceptance': False}}
    finally:
        diffusion.guide = original_guide
        model.train()
        model.clip_model.eval()


def smoke(cfg, data, stats, provenance, device):
    model, init_receipt = build_model(cfg)
    model.to(device)
    embeddings = text_cache(model, data, device)
    diffusion = diffusion_for(stats)
    train = [r for r in data if r['row']['split'] == 'train']
    chosen = sorted(train, key=lambda x: len(x['motion']))[:4] + sorted(train, key=lambda x: len(x['motion']))[-4:]
    motion, cond = batch_from(chosen, stats, embeddings, device)
    model.eval()
    t = torch.full((len(motion),), 500, device=device, dtype=torch.long)
    with torch.no_grad():
        before = model(motion, t, **cond)
        changed = motion.clone()
        changed.masked_fill_(~cond['y']['mask'].expand_as(changed), 1234.)
        after = model(changed, t, **cond)
        padding_delta = float(((before - after) * cond['y']['mask']).abs().max())
    require(padding_delta < 1e-5, 'padded values affected valid model frames')
    initial_zero = model.zero_convs[0].weight.detach().clone()
    initial_hint = model.input_hint_block.poseEmbedding[-1].weight.detach().clone()
    model.train()
    model.clip_model.eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    losses, times = [], []
    for _ in range(8):
        began = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, diffusion, motion, cond)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        require(torch.isfinite(norm), 'nonfinite gradients')
        optimizer.step()
        torch.cuda.synchronize(device)
        losses.append(float(loss))
        times.append(time.monotonic() - began)
    zero_change = float((model.zero_convs[0].weight - initial_zero).abs().max())
    hint_change = float((model.input_hint_block.poseEmbedding[-1].weight - initial_hint).abs().max())
    require(zero_change > 0 and hint_change > 0, 'control and hint networks did not train')
    cursor = {'step': 8, 'epoch': 0, 'next_batch': 0}
    data_to_save = payload(model, optimizer, cursor, cfg, provenance, device)
    buffer = io.BytesIO()
    torch.save(data_to_save, buffer)
    size = buffer.tell()
    require(size < cfg['checkpoint_size_budget'], 'actual checkpoint exceeds budget')
    buffer.seek(0)
    readback = torch.load(buffer, map_location='cpu', weights_only=True)
    require(readback['cursor'] == cursor and readback['provenance'] == provenance, 'checkpoint metadata roundtrip')
    require(all(torch.equal(v, readback['model'][k]) for k, v in data_to_save['model'].items()), 'checkpoint model roundtrip')
    out = Path(cfg['stage_dir']) / 'reports'
    probe_item = next(r for r in data if r['row']['split'] == 'val' and r['row']['dataset_source'] == 'OmniRetarget/OMOMO')
    sampled = sampling_probe(model, diffusion, probe_item, stats, embeddings, device)
    with (out / 'smoke_sample.npz').open('xb') as f:
        np.savez_compressed(f, features=sampled['features'], canonical_points=sampled['canonical_points'],
                            target_points=sampled['target_points'], target_mask=sampled['target_mask'],
                            metadata_utf8=np.frombuffer(json.dumps(sampled['metadata']).encode(), dtype=np.uint8))
    report = {'model_initialization': init_receipt, 'provenance': provenance,
              'real_gpu_optimizer_steps': 8, 'losses': losses, 'microbatch_size': len(motion),
              'steady_microbatch_seconds': sum(times[2:]) / len(times[2:]),
              'valid_frame_padding_max_difference': padding_delta,
              'zero_conv_weight_change': zero_change, 'hint_weight_change': hint_change,
              'real_adam_checkpoint_serialized_bytes': size, 'checkpoint_roundtrip': True,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'sampling': sampled['metadata'], 'scope': 'pipeline witness, not a converged model'}
    write_json(out / 'gpu_smoke.json', report, exclusive=True)
    print(json.dumps(report), flush=True)


def train(cfg, data, stats, provenance, device, resume=None):
    stage = Path(cfg['stage_dir'])
    smoke_report = json.loads((stage / 'reports/gpu_smoke.json').read_text())
    require(smoke_report['checkpoint_roundtrip'] and smoke_report['real_gpu_optimizer_steps'] == 8, 'real training smoke required')
    estimated = cfg['checkpoint_size_budget']
    planned = 3  # early checkpoint + midpoint + final; keep all, no automatic deletion.
    need = max(11 * 1024**3, cfg['disk_floor_bytes'] + (planned + 2) * estimated)
    require(shutil.disk_usage(stage).free >= need, 'planned training/checkpoint disk budget')
    run = stage / 'runs' / cfg['run_name']
    if resume is None:
        run.mkdir(exist_ok=False)
    else:
        require(run.is_dir() and Path(resume).resolve().parent == run.resolve(), 'resume belongs to this run')
    lock = run / 'training.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    try:
        model, receipt = build_model(cfg)
        model.to(device)
        embeddings = text_cache(model, data, device)
        diffusion = diffusion_for(stats)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        cursor = {'step': 0, 'epoch': 0, 'next_batch': 0}
        if resume:
            saved = torch.load(resume, map_location='cpu', weights_only=True)
            require(saved['provenance'] == provenance and saved['config'] == cfg, 'resume provenance/config mismatch')
            restore_state(model, saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            cursor = saved['cursor']
            restore_rng(saved['rng'], device)
        else:
            write_json(run / 'config.json', cfg, exclusive=True)
            write_json(run / 'provenance.json', {**provenance, 'model_initialization': receipt}, exclusive=True)
        training = [r for r in data if r['row']['split'] == 'train']
        validation = [r for r in data if r['row']['split'] == 'val']
        seed_all(cfg['seed']) if not resume else None
        initial_val = evaluate(model, diffusion, validation, stats, embeddings, device)
        model.train()
        model.clip_model.eval()
        started = time.monotonic()
        last_checkpoint = None
        status = {'status': 'running', 'pid': os.getpid(), 'device': str(device), 'cursor': cursor,
                  'initial_validation_denoising_mse': initial_val, 'last_checkpoint': None}
        write_json(run / 'status.json', status)
        with (run / 'metrics.jsonl').open('a' if resume else 'x', buffering=1) as log:
            log.write(json.dumps({'event': 'initial_validation', 'step': cursor['step'], 'mse': initial_val}) + '\n')
            while cursor['step'] < cfg['max_steps']:
                update_started = time.monotonic()
                micro = []
                for _ in range(cfg['gradient_accumulation']):
                    g = torch.Generator().manual_seed(cfg['seed'] + cursor['epoch'])
                    order = torch.randperm(len(training), generator=g).tolist()
                    batches = math.ceil(len(order) / cfg['microbatch_size'])
                    start = cursor['next_batch'] * cfg['microbatch_size']
                    selected = [training[i] for i in order[start:start + cfg['microbatch_size']]]
                    micro.append(batch_from(selected, stats, embeddings, device))
                    cursor['next_batch'] += 1
                    if cursor['next_batch'] == batches:
                        cursor['next_batch'] = 0
                        cursor['epoch'] += 1
                        break
                total = sum(len(x) for x, _ in micro)
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.
                for motion, condition in micro:
                    loss = loss_for(model, diffusion, motion, condition)
                    (loss * (len(motion) / total)).backward()
                    total_loss += float(loss) * len(motion) / total
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                require(torch.isfinite(norm), 'nonfinite training gradients')
                optimizer.step()
                cursor['step'] += 1
                torch.cuda.synchronize(device)
                metrics = {'step': cursor['step'], 'epoch': cursor['epoch'], 'next_batch': cursor['next_batch'],
                           'loss': total_loss, 'samples': total, 'gradient_norm': float(norm),
                           'update_seconds': time.monotonic() - update_started,
                           'elapsed_s': time.monotonic() - started,
                           'gpu_allocated_bytes': torch.cuda.memory_allocated(device)}
                if cursor['step'] % cfg['validation_every'] == 0:
                    metrics['validation_denoising_mse'] = evaluate(model, diffusion, validation, stats, embeddings, device)
                should_save = cursor['step'] in (cfg['early_checkpoint_step'], cfg['max_steps']) or cursor['step'] % cfg['checkpoint_every'] == 0
                if should_save:
                    last_checkpoint = checkpoint(run / ('step_%09d.pt' % cursor['step']),
                                                 payload(model, optimizer, dict(cursor), cfg, provenance, device), cfg)
                    metrics['checkpoint'] = last_checkpoint
                log.write(json.dumps(metrics, allow_nan=False) + '\n')
                if cursor['step'] % 10 == 0 or should_save:
                    status.update(cursor=dict(cursor), last_metrics=metrics, last_checkpoint=last_checkpoint)
                    write_json(run / 'status.json', status)
                    print(json.dumps(metrics), flush=True)
                if cursor['step'] % 25 == 0:
                    require(shutil.disk_usage(stage).free >= cfg['disk_floor_bytes'] + estimated, 'disk reserve reached; preserve existing checkpoints')
            status.update(status='completed', cursor=dict(cursor), last_checkpoint=last_checkpoint)
            write_json(run / 'status.json', status)
            print(json.dumps(status), flush=True)
    except Exception as exc:
        write_json(run / 'failure.json', {'error': type(exc).__name__ + ': ' + str(exc), 'pid': os.getpid()})
        raise
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=['verify-env', 'preflight', 'smoke', 'train'], required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root, device = setup(cfg)
    if args.mode == 'verify-env':
        gen = torch.Generator(device=device).manual_seed(17)
        x = torch.randn((16, 16), device=device, generator=gen)
        out = x @ x.T
        require(torch.isfinite(out).all(), 'CUDA kernel witness failed')
        print('WITNESS', json.dumps({'shape': list(out.shape), 'device': str(device), 'torch': torch.__version__,
                                     'gpu': torch.cuda.get_device_name(device), 'seed': 17,
                                     'memory_fraction': cfg['gpu_memory_fraction'], 'checksum': float(out.sum())}), flush=True)
        return
    data, stats, provenance = preflight(cfg)
    if args.mode == 'preflight':
        write_json(Path(cfg['stage_dir']) / 'reports/preflight.json', provenance, exclusive=True)
        print(json.dumps(provenance), flush=True)
    elif args.mode == 'smoke':
        smoke(cfg, data, stats, provenance, device)
    else:
        train(cfg, data, stats, provenance, device, args.resume)


if __name__ == '__main__':
    main()
