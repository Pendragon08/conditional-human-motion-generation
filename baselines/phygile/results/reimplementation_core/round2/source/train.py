"""Independent generator training entrypoint. Requires prepared, audited 262D/text data.

This is a reconstruction choice, not the unpublished author's training program.
Synthetic inputs are accepted only when both manifest and config say unit_test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from features import BlockNormalizer, asfo_multipliers
from model import ModelConfig, MotionDenoiser
from text import FrozenDistilBert
from diffusion import X0Diffusion

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_manifest(path, purpose):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest['format'] != 'phygile262-explicit-v1' or manifest['purpose'] != purpose:
        raise ValueError('Manifest format/purpose differs from the requested run')
    if purpose not in ['research_training', 'unit_test']:
        raise ValueError('Unknown run purpose')
    for field in ['feature_semantics', 'source_and_license', 'mapping_status']:
        if not manifest.get(field):
            raise ValueError('Explicit data semantics and provenance are required: ' + field)
    if manifest['mapping_status'] not in ['author_verified', 'explicit_reconstruction', 'synthetic_unit_test']:
        raise ValueError('Unknown mapping status')
    if manifest['mapping_status'] == 'synthetic_unit_test' and purpose != 'unit_test':
        raise ValueError('Synthetic tensors cannot be labeled research training')
    if not manifest['splits'].get('train'):
        raise ValueError('No paired training sequences')
    groups, hashes, training = {}, {}, []
    for split, entries in manifest['splits'].items():
        groups[split] = set(); hashes[split] = set()
        for entry in entries:
            source = (path.parent / entry['features']).resolve(strict=True)
            sha = digest(source)
            if sha != entry['sha256']:
                raise ValueError('Motion input hash differs: ' + str(source))
            if not entry.get('source_group') or not entry.get('texts') or not all(isinstance(x,str) and x.strip() for x in entry['texts']):
                raise ValueError('Each motion needs a source group and real text annotations')
            if not entry.get('annotation_source'):
                raise ValueError('Text annotation provenance is required')
            groups[split].add(entry['source_group']); hashes[split].add(sha)
            if split == 'train':
                values = np.load(source, allow_pickle=False)
                if values.ndim != 2 or values.shape[1] != 262 or len(values) < 2 or not np.isfinite(values).all():
                    raise ValueError('Expected finite [frames,262] features')
                if not np.isin(values[:,256:], [0,1]).all():
                    raise ValueError('Contact channels must be binary before normalization')
                if np.max(np.abs(values[:,43:217])) > 1.0001:
                    raise ValueError('6D rotation channels must be in [-1,1]')
                training.append((values.astype(np.float32), entry))
    for left in groups:
        for right in groups:
            if left < right and (groups[left] & groups[right] or hashes[left] & hashes[right]):
                raise ValueError('Source-group or file overlap across splits')
    return manifest, training

class PairedMotionData(Dataset):
    def __init__(self, training, normalizer, asfo):
        self.items = [(normalizer.transform(x), item) for x,item in training]
        counts = np.ones(len(training), dtype=np.int64)
        if asfo['enabled']:
            counts, self.asfo_metadata = asfo_multipliers([item['tags'] for _,item in training], asfo['rho_max'])
            if asfo['count_policy'] == 'keep_at_least_one':
                counts = np.maximum(counts, 1)
            elif asfo['count_policy'] != 'literal_allow_zero':
                raise ValueError('Choose and document an ASFO zero-count policy')
        else:
            self.asfo_metadata = {'enabled':False}
        if asfo['mirroring']:
            raise ValueError('Mirroring is not implemented; cannot silently enable it')
        self.indices = np.repeat(np.arange(len(self.items)), counts).tolist()
        if not self.indices:
            raise ValueError('ASFO excluded all training data')
        self.asfo_metadata['effective_counts'] = counts.tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        motion, item = self.items[self.indices[index]]
        return motion, random.choice(item['texts'])

def collate(batch):
    length = max(len(x) for x,_ in batch)
    values = torch.zeros(len(batch), length, 262)
    valid = torch.zeros(len(batch), length, dtype=torch.bool)
    for i,(motion,_) in enumerate(batch):
        values[i,:len(motion)] = torch.from_numpy(motion)
        valid[i,:len(motion)] = True
    return values, valid, [text for _,text in batch]

def train(config_path, output, device):
    config_path = Path(config_path).resolve(); output = Path(output)
    config = json.loads(config_path.read_text())
    manifest_path = (config_path.parent/config['dataset_manifest']).resolve()
    manifest, training = load_manifest(manifest_path, config['purpose'])
    if config['max_updates'] < 1 or config['batch_size'] < 1 or config['save_every'] < 1:
        raise ValueError('Positive update, batch and checkpoint intervals required')
    if max(len(x) for x,_ in training) > config['max_sequence_frames']:
        raise ValueError('Sequence exceeds declared budget; refusing caption-misaligned silent cropping')
    normalizer = BlockNormalizer.fit([x for x,_ in training])
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    dataset = PairedMotionData(training, normalizer, config['asfo'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        collate_fn=collate, num_workers=0, generator=torch.Generator().manual_seed(seed))
    encoder_path = (config_path.parent/config['text_encoder_path']).resolve(strict=True)
    text_provenance = json.loads((encoder_path/'download.json').read_text())
    for entry in text_provenance['files']:
        if digest(encoder_path/entry['name']) != entry['sha256']:
            raise ValueError('Text encoder file differs from pinned provenance')
    encoder = FrozenDistilBert(encoder_path).to(device)
    model = MotionDenoiser(ModelConfig(**config['model'])).to(device)
    schedule = config['schedule']
    if schedule['type'] != 'linear':raise ValueError('This entrypoint supports an explicitly chosen linear schedule only')
    diffusion = X0Diffusion(model, torch.linspace(schedule['start'], schedule['end'], schedule['steps'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  betas=tuple(config['adam_betas']), weight_decay=config['weight_decay'])
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output/'normalization.npz', mean=normalizer.mean, std=normalizer.std)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    record = {'status':'RUNNING', 'purpose':config['purpose'], 'author_training_program':False,
              'manifest_sha256':digest(manifest_path), 'config_sha256':digest(config_path),
              'mapping_status':manifest['mapping_status'], 'text_encoder_revision':text_provenance['revision'],
              'training_sequences':len(training), 'training_frames':sum(len(x) for x,_ in training),
              'asfo':dataset.asfo_metadata, 'device':str(device), 'torch':torch.__version__,
              'source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
              'motion_quality_evaluated':False, 'physics_loop_run':False}
    started = time.monotonic(); step = 0; model.train()
    try:
        with (output/'training_losses.jsonl').open('w') as log:
            while step < config['max_updates']:
                for clean, frames, texts in loader:
                    clean, frames = clean.to(device), frames.to(device)
                    tokens, text_valid = encoder(texts)
                    losses = diffusion.training_loss(clean,tokens,text_valid,frames,
                        condition_dropout=config['condition_dropout'], balance_weight=config['balance_weight'])
                    if not torch.isfinite(losses['loss']):raise FloatingPointError('Nonfinite diffusion training loss')
                    optimizer.zero_grad(set_to_none=True); losses['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step(); step += 1
                    row = {'update':step, **{key:float(value.detach()) for key,value in losses.items()}}
                    log.write(json.dumps(row)+'\n'); log.flush()
                    if step % config['save_every'] == 0 or step == config['max_updates']:
                        checkpoint = {'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),
                            'optimizer_updates':step,'config':config,'normalization_mean':normalizer.mean,
                            'normalization_std':normalizer.std,'manifest_sha256':record['manifest_sha256'],
                            'text_encoder_revision':text_provenance['revision'],'purpose':config['purpose']}
                        torch.save(checkpoint, output/f'model_{step:06d}.pt')
                    if step >= config['max_updates']:break
        record['status'] = 'COMPLETED'
    except BaseException as error:
        record['status'] = 'FAILED'; record['error'] = type(error).__name__+': '+str(error); raise
    finally:
        record['completed_updates'] = step; record['elapsed_seconds'] = time.monotonic()-started
        (output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    return record

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    train(args.config, args.output, torch.device(args.device))
