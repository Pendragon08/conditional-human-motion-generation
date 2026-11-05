"""Experimental, provenance-explicit robot motion loader for OmniControl.

Consumes audited (T,263) feature files with explicit captions and group/split
metadata. It does not generate or validate retargeted motions or infer labels.
The resulting collate format includes an explicit 22-joint hint mask.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from grouped_split import audit_assignments


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _stats(value, size: int, *, positive: bool = False) -> torch.Tensor:
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(a) != size or not np.isfinite(a).all() or (positive and np.any(a <= 0)):
        raise ValueError(f'invalid normalization statistics: expected {size} finite entries')
    t = torch.as_tensor(a.copy(), dtype=torch.float32)
    if not torch.isfinite(t).all() or (positive and torch.any(t <= 0)):
        raise ValueError('normalization statistics cannot be represented as finite float32')
    return t


class RobotMotionDataset(Dataset):
    """Read explicitly assigned clips; verify provenance and hashes up front.

    `records` is the full manifest, including all splits; group leakage is
    checked before filtering. Caller must supply separately audited train-only
    normalization values; this class does not establish their provenance.
    """
    def __init__(self, *, root: Path, records: list[dict], split: str,
                 feature_mean, feature_std, spatial_mean, spatial_std,
                 min_frames: int = 2, max_frames: int = 199):
        if split not in ('train', 'val', 'test') or not isinstance(records, list) or not records:
            raise ValueError('a valid split and nonempty full manifest are required')
        if not isinstance(min_frames, int) or not isinstance(max_frames, int) or not (2 <= min_frames <= max_frames):
            raise ValueError('invalid frame limits')
        audit_assignments(records)
        self.root = Path(root).resolve(strict=True)
        self.mean = _stats(feature_mean, 263); self.std = _stats(feature_std, 263, positive=True)
        self.spatial_mean = _stats(spatial_mean, 66).reshape(22, 3)
        self.spatial_std = _stats(spatial_std, 66, positive=True).reshape(22, 3)
        self.min_frames, self.max_frames = min_frames, max_frames
        self.records = []
        for record in records:
            if record['split'] != split:
                continue
            caption, source, relative, checksum = (record.get('caption'), record.get('caption_source'),
                                                    record.get('feature_path'), record.get('feature_sha256'))
            if not all(isinstance(x, str) and x.strip() for x in (caption, source, relative, checksum)):
                raise ValueError('each clip needs caption, caption_source, feature_path, and feature_sha256')
            relative_path = Path(relative)
            if relative_path.is_absolute():
                raise ValueError('feature path must be relative to the dataset root')
            path = (self.root / relative_path).resolve(strict=True)
            if not path.is_relative_to(self.root) or not path.is_file() or path.suffix != '.npz':
                raise ValueError('feature path must stay inside dataset root and name an NPZ file')
            if len(checksum) != 64 or _digest(path) != checksum.lower():
                raise ValueError(f'feature hash mismatch: {record["sample_id"]}')
            stat = path.stat()
            self.records.append({**record, '_path': path, '_file_size': stat.st_size, '_file_mtime_ns': stat.st_mtime_ns})
        if not self.records:
            raise ValueError(f'no clips in split {split!r}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from data_loaders.humanml.scripts.motion_process import recover_from_ric
        record = self.records[index]
        stat = record['_path'].stat()
        if stat.st_size != record['_file_size'] or stat.st_mtime_ns != record['_file_mtime_ns']:
            raise RuntimeError(f'feature file changed after validation: {record["sample_id"]}')
        with np.load(record['_path'], allow_pickle=False) as data:
            features = np.asarray(data['features'], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 263 or not self.min_frames <= len(features) <= self.max_frames or not np.isfinite(features).all():
            raise ValueError(f'invalid feature array for {record["sample_id"]}')
        raw = torch.from_numpy(features.copy())
        with torch.no_grad():
            joints = recover_from_ric(raw, 22)
        if joints.shape != (len(features), 22, 3) or not torch.isfinite(joints).all():
            raise ValueError(f'could not recover finite keypoints for {record["sample_id"]}')
        # Match the available first-generation code's one-joint, random-frame
        # policy. It is an explicit approximation pending v2 implementation.
        choices = torch.tensor([0, 10, 11, 15, 20, 21])
        joint = int(choices[torch.randint(len(choices), (1,))])
        count = int(torch.randint(1, len(features), (1,)))
        times = torch.randperm(len(features))[:count]
        mask = torch.zeros((len(features), 22), dtype=torch.bool)
        mask[times, joint] = True
        hint = ((joints - self.spatial_mean) / self.spatial_std) * mask[..., None]
        normalized = (raw-self.mean)/self.std
        if not torch.isfinite(hint).all() or not torch.isfinite(normalized).all():
            raise ValueError(f'normalization produced non-finite values for {record["sample_id"]}')
        return {'sample_id': record['sample_id'], 'text': record['caption'],
                'motion': normalized, 'hint': hint.reshape(len(features), 66),
                'hint_mask': mask}


def collate_robot_motion(items: list[dict]):
    if not items:
        raise ValueError('empty batch')
    batch = len(items)
    lengths = torch.as_tensor([len(item['motion']) for item in items], dtype=torch.long)
    max_length = int(lengths.max())
    motion = torch.zeros((batch, 263, 1, max_length), dtype=torch.float32)
    hint = torch.zeros((batch, max_length, 66), dtype=torch.float32)
    hint_mask = torch.zeros((batch, max_length, 22), dtype=torch.bool)
    for i, item in enumerate(items):
        n = int(lengths[i]); motion[i, :, 0, :n] = item['motion'].T
        hint[i, :n] = item['hint']; hint_mask[i, :n] = item['hint_mask']
    valid = torch.arange(max_length)[None, :] < lengths[:, None]
    return motion, {'y': {'mask': valid[:, None, None, :], 'lengths': lengths,
                          'text': [item['text'] for item in items], 'hint': hint,
                          'hint_mask': hint_mask, 'seq_name': [item['sample_id'] for item in items]}}
