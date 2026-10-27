"""Read actual prepared pairs through the training Dataset/DataLoader, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'reimplementation'))
from train import load_manifest, PairedMotionData, collate
from features import BlockNormalizer


def check(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    _, training = load_manifest(a.manifest, 'research_training')
    with np.load(a.manifest.parent / 'normalization_train_only.npz') as n:
        normalizer = BlockNormalizer(n['mean'], n['std'])
    dataset = PairedMotionData(training, normalizer,
        {'enabled': False, 'mirroring': False})
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
        collate_fn=collate, num_workers=0)
    frames = sequences = batches = 0
    for values, valid, texts in loader:
        if values.ndim != 3 or values.shape[-1] != 262 or valid.shape != values.shape[:2]:
            raise ValueError('Unexpected collated shape')
        if not torch.isfinite(values).all() or not torch.isin(values[valid][:, 256:], torch.tensor([0., 1.])).all():
            raise ValueError('Nonfinite values or invalid contacts')
        if len(texts) != len(values) or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Missing caption')
        sequences += len(values); frames += int(valid.sum()); batches += 1
    if sequences != len(training) or frames != sum(len(x) for x, _ in training):
        raise ValueError('DataLoader changed input coverage')
    result = {'status': 'PAIRED_BATCH_LOADING_PASSED', 'training_sequences': sequences,
        'valid_training_frames': frames, 'batches_read': batches, 'batch_size': 4,
        'manifest_function_called': True, 'paired_motion_dataset_instantiated': True,
        'pytorch_dataloader_iterated': True, 'all_padded_tensors_finite': True,
        'valid_contact_channels_binary': True, 'online_asfo_enabled': False,
        'online_mirroring_enabled': False, 'generator_instantiated': False,
        'optimizer_updates': 0, 'motion_quality_evaluated': False,
        'device': 'cpu', 'seed': 42,
        'manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_module_sha256': hashlib.sha256((BASE/'reimplementation/train.py').read_bytes()).hexdigest()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    check(p.parse_args())
