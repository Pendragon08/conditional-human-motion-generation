"""Experimental exclusive checkpoint IO; does not implement a training runner.

Data cursor metadata is saved, but exact resumption additionally depends on the
caller restoring the same sampler, batches, worker settings and environment.
Only load checkpoints from trusted runs with verified provenance.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import numpy as np
import torch

_REQUIRED = ('dataset_manifest_sha256','stats_sha256','model_init_sha256')


def _provenance(value):
    if not isinstance(value, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in value.items()):
        raise ValueError('provenance must be a dictionary of strings')
    if any(not re.fullmatch('[0-9a-f]{64}', value.get(key,'')) for key in _REQUIRED):
        raise ValueError('dataset, stats and initialization SHA-256 values are required')
    return dict(sorted(value.items()))


def _cursor(value):
    if not isinstance(value,dict) or set(value) != {'step','epoch','next_batch'}:
        raise ValueError('cursor must contain step, epoch, and next_batch')
    if any(isinstance(n,bool) or not isinstance(n,int) or n < 0 for n in value.values()):
        raise ValueError('cursor fields must be nonnegative integers')
    return dict(value)


def _rng_state():
    name, values, pos, has_gauss, cached = np.random.get_state()
    return {'python':random.getstate(), 'numpy':[name,values.tolist(),int(pos),int(has_gauss),float(cached)],
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


class _BoundedWriter:
    """Keep an upper bound and check current free space before each write.

    This does not reserve filesystem capacity against other processes.
    """
    def __init__(self, file, directory, limit, floor):
        self.file, self.directory, self.limit, self.floor = file, directory, limit, floor

    def write(self, data):
        size = len(data)
        if self.file.tell() + size > self.limit:
            raise RuntimeError('checkpoint exceeds size estimate')
        view = memoryview(data)
        written = 0
        while written < size:
            if shutil.disk_usage(self.directory).free < self.floor + (size - written):
                raise RuntimeError('insufficient free disk during checkpoint write')
            count = self.file.write(view[written:])
            if count is None or count <= 0:
                raise OSError('checkpoint write made no progress')
            written += count
        return written

    def __getattr__(self, name):
        return getattr(self.file, name)


def save_checkpoint(path: Path, model, optimizer, *, provenance: dict, config: dict, cursor: dict,
                    max_checkpoint_bytes=None, disk_floor_bytes=None):
    path = Path(path)
    provenance, cursor = _provenance(provenance), _cursor(cursor)
    # Restrict config to JSON values, retaining a canonical copy.
    config = json.loads(json.dumps(config,sort_keys=True,allow_nan=False))
    if not isinstance(config,dict): raise ValueError('config must be a JSON object')
    if not path.parent.is_dir(): raise ValueError('checkpoint directory must exist')
    if (max_checkpoint_bytes is None) != (disk_floor_bytes is None):
        raise ValueError('checkpoint size and disk floor must be supplied together')
    if max_checkpoint_bytes is not None:
        if any(isinstance(x,bool) or not isinstance(x,int) or x <= 0 for x in (max_checkpoint_bytes,disk_floor_bytes)):
            raise ValueError('checkpoint size and disk floor must be positive integers')
        if shutil.disk_usage(path.parent).free < disk_floor_bytes + max_checkpoint_bytes:
            raise RuntimeError('insufficient free disk before checkpoint write')
    payload = {'version':1,'provenance':provenance,'config':config,'cursor':cursor,
               'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':_rng_state()}
    temp = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.checkpoint-',suffix='.tmp',dir=path.parent,delete=False,buffering=0) as f:
            temp = Path(f.name)
            writer = _BoundedWriter(f,path.parent,max_checkpoint_bytes,disk_floor_bytes) if max_checkpoint_bytes is not None else f
            torch.save(payload,writer)
            f.flush()
            os.fsync(f.fileno())
        if disk_floor_bytes is not None and shutil.disk_usage(path.parent).free < disk_floor_bytes:
            raise RuntimeError('insufficient free disk after checkpoint write')
        # Atomic publish without replacing any existing file; same filesystem.
        os.link(temp,path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def load_checkpoint(path: Path, model, optimizer, *, expected_provenance: dict, expected_config: dict):
    expected = _provenance(expected_provenance)
    config = json.loads(json.dumps(expected_config,sort_keys=True,allow_nan=False))
    payload = torch.load(Path(path),map_location='cpu',weights_only=True)
    if not isinstance(payload,dict) or payload.get('version') != 1:
        raise ValueError('unknown checkpoint format')
    if payload.get('provenance') != expected or payload.get('config') != config:
        raise ValueError('checkpoint provenance or configuration mismatch')
    cursor = _cursor(payload.get('cursor'))
    rng = payload['rng']
    if rng['cuda'] and (not torch.cuda.is_available() or torch.cuda.device_count() != len(rng['cuda'])):
        raise ValueError('CUDA RNG topology differs from checkpoint')
    model.load_state_dict(payload['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    random.setstate(rng['python'])
    name, values, pos, has_gauss, cached = rng['numpy']
    np.random.set_state((name,np.asarray(values,dtype=np.uint32),pos,has_gauss,cached))
    torch.set_rng_state(rng['torch'])
    if rng['cuda']: torch.cuda.set_rng_state_all(rng['cuda'])
    return cursor
