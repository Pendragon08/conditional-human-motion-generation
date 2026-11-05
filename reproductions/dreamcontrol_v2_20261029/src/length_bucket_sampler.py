"""Experimental exact-length batches for an unmasked attention backbone.

DreamControl's upstream CMDM transformer does not use its valid-frame mask in
self-attention. Mixed lengths therefore allow padded/noisy frames to influence
valid predictions despite masked losses. Equal-length batches avoid introducing
padding within a batch, at the cost of small batches and different sampling.
This does not alter or prove fidelity of the upstream architecture.
"""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
import torch
from torch.utils.data import Sampler


class ExactLengthBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, *, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise ValueError('shuffle and drop_last must be boolean')
        self.batch_size, self.shuffle, self.seed, self.drop_last = batch_size, shuffle, seed, drop_last
        self.epoch = 0
        groups = defaultdict(list)
        for index, record in enumerate(dataset.records):
            path = record['_path']
            stat = path.stat()
            if stat.st_size != record['_file_size'] or stat.st_mtime_ns != record['_file_mtime_ns']:
                raise RuntimeError('feature file changed after dataset validation')
            with np.load(path, allow_pickle=False) as data:
                features = data['features']
            if features.ndim != 2 or features.shape[1] != 263 or not dataset.min_frames <= len(features) <= dataset.max_frames:
                raise ValueError('invalid feature shape or frame count')
            groups[len(features)].append(index)
        if not groups:
            raise ValueError('dataset has no records')
        self.groups = dict(sorted(groups.items()))

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError('epoch must be a nonnegative integer')
        self.epoch = epoch

    def __len__(self):
        fn = lambda n: n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        return sum(fn(len(indices)) for indices in self.groups.values())

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for indices in self.groups.values():
            order = torch.randperm(len(indices), generator=generator).tolist() if self.shuffle else range(len(indices))
            ordered = [indices[i] for i in order]
            for start in range(0, len(ordered), self.batch_size):
                batch = ordered[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        order = torch.randperm(len(batches), generator=generator).tolist() if self.shuffle else range(len(batches))
        for i in order:
            yield batches[i]
