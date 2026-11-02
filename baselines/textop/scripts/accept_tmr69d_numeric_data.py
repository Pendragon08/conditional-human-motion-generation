"""Actually rehash/decode accepted TRAIN NPZ and encode the frozen 69D codec.

Keeps one full source in RAM. No encoder weights, SDK model or optimization.
Normalized feature bytes are hashed, not persisted or published.
"""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import time
import traceback
from ldm100k_verified_input import ROOT, read, sha, owned

SHM = pathlib.Path('/dev/shm/ouyanghao_textop_20261025')


def own_input(p):
    path = pathlib.Path(p)
    roots = [x for x in [ROOT, SHM] if path == x or x in path.parents]
    assert len(roots) == 1
    return owned(path, roots[0])


p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['executor_sha256']
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and set(os.sched_getaffinity(0)) == {0, 1, 2, 3}
rp = owned(s['scope_receipt']['path'])
assert sha(rp) == s['scope_receipt']['sha256']
scope_receipt = read(rp)
assert scope_receipt['status'] == 'TMR69D_TRAINING_METADATA_SCOPE_BUILT' and scope_receipt['output_writers_closed']
item = scope_receipt['private_artifact']
dp = owned(item['path'])
assert dp.stat().st_size == item['bytes'] and sha(dp) == item['sha256']
data = read(dp)
assert data['summary']['original_train_families'] == 5361
assert sha(ROOT / 'scripts/paper69_runtime.py') == s['codec_sha256']
parent = owned(ROOT / 'evaluator_tmr_20261029')
out = parent / s['run']
out.mkdir(mode=0o700, exist_ok=False)
started = time.monotonic()
receipt = dict(status='STARTING_TMR69D_NUMERIC_DATA_ACCEPTANCE', run=s['run'], source_sha256=sha(__file__),
    request_sha256=sha(sp), output_writers_closed=False, processed_sources=0, encoder_weight_loads=0,
    sdk_simulation_model_constructions=0, physics_steps=0, optimizer_updates=0, evaluator_training_authorized=False)


def save():
    receipt['elapsed_seconds'] = time.monotonic() - started
    tmp = out / 'receipt.partial'
    with tmp.open('w') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(out / 'receipt.json')


save()
try:
    import numpy as np
    import torch
    import paper69_runtime as codec
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    norm = own_input(data['normalization']['path'])
    assert sha(norm) == data['normalization']['sha256']
    mean, std = torch.load(norm, map_location='cpu')
    assert mean.shape == std.shape == (69,) and mean.dtype == std.dtype == torch.float32
    assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()
    samples_by_source = {}
    for x in data['samples']:
        samples_by_source.setdefault(x['canonical_source'], []).append(x)
    assert set(samples_by_source) == {x['canonical_source'] for x in data['sources']}
    records, probes = [], []
    total_features = 0
    for row in data['sources']:
        assert row['original_split'] == 'train' and row['mirrored'] is False
        path = own_input(row['output_file'])
        assert sha(path) == row['output_sha256']
        before = path.stat()
        with np.load(path, allow_pickle=False) as motion:
            q = np.array(motion['qpos_29'], dtype=np.float32)
            contact = np.array(motion['contact'], dtype=np.float32)
        assert q.shape == (row['frames50'], 36) and contact.shape == (row['frames50'], 2)
        assert np.isfinite(q).all() and np.isfinite(contact).all()
        assert np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-3)
        assert ((contact >= 0) & (contact <= 1)).all()
        with torch.inference_mode():
            features, _ = codec.encode69(dict(root_trans_offset=torch.from_numpy(q[:, :3]),
                root_rot=torch.from_numpy(q[:, [4, 5, 6, 3]]), dof=torch.from_numpy(q[:, 7:]), contact_mask=torch.from_numpy(contact)))
            assert features.shape == (row['frames50'] - 1, 69) and features.dtype == torch.float32 and torch.isfinite(features).all()
            normalized = ((features - mean) / std).contiguous()
            assert torch.isfinite(normalized).all()
        for sample in samples_by_source[row['canonical_source']]:
            assert sample['split'] == row['split'] and sample['original_split'] == 'train'
            piece = normalized[sample['start_feature']:sample['end_feature_exclusive']]
            assert piece.shape == (sample['feature_frames'], 69) and torch.isfinite(piece).all()
            if row['split'] == 'gradient_train' and len(probes) < 32:
                probes.append(piece.clone())
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) == (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev)
        records.append(dict(canonical_source=row['canonical_source'], split=row['split'], path=str(path),
            source_sha256=row['output_sha256'], source_bytes=before.st_size, pose_frames=len(q), feature_frames=len(normalized),
            feature_sha256=hashlib.sha256(features.numpy().tobytes()).hexdigest(),
            normalized_feature_sha256=hashlib.sha256(normalized.numpy().tobytes()).hexdigest(),
            normalized_feature_bytes=normalized.numel()*4, finite=True, unit_quaternion=True))
        total_features += len(normalized)
        receipt['processed_sources'] = len(records)
        if len(records) % 100 == 0:
            save()
        assert shutil.disk_usage(out).free > 2 * 1024**3 + 128 * 1024**2
    assert len(records) == 5361 and len(probes) == 32
    maxlen = max(len(x) for x in probes)
    padded = torch.zeros(32, maxlen, 69)
    mask = torch.zeros(32, maxlen, dtype=torch.bool)
    for i, value in enumerate(probes):
        padded[i, :len(value)] = value
        mask[i, :len(value)] = True
    assert torch.isfinite(padded).all() and torch.equal(padded[~mask], torch.zeros_like(padded[~mask]))
    assert not torch.cuda.is_initialized() and sha(norm) == data['normalization']['sha256'] and sha(dp) == item['sha256']
    raw = json.dumps(records, indent=2, allow_nan=False).encode()
    assert len(raw) < s['output_upper_bytes']
    artifact = out / 'numeric_sources.private.json'
    with artifact.open('xb') as f:
        os.chmod(artifact, 0o600)
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    receipt.update(status='TMR69D_NUMERIC_TRAIN_INPUTS_ACCEPTED', output_writers_closed=True,
        original_train_sources=5361, original_validation_sources_read=0, actual_pose_frames=sum(x['pose_frames'] for x in records),
        actual_unique_source_feature_frames=total_features, actual_source_npz_bytes=sum(x['source_bytes'] for x in records),
        full_normalized_feature_RAM_bytes=total_features*69*4, normalized_features_persisted=False,
        actual_sample_windows_verified=len(data['samples']), maximum_source_RAM_pose_bytes=max(x['pose_frames']*38*4 for x in records),
        variable_length_probe=dict(batch_shape=list(padded.shape), mask_shape=list(mask.shape), finite=True, padded_exact_zero=True,
            source_slice_sha256=hashlib.sha256(padded.numpy().tobytes()).hexdigest()),
        private_artifact=dict(path=str(artifact),bytes=len(raw),sha256=sha(artifact)),
        scope_receipt_sha256=sha(rp), scope_artifact_sha256=sha(dp), codec_sha256=s['codec_sha256'], normalization_sha256=sha(norm),
        cuda_initialized=False, mujoco_library_imported=True, complete_evaluator_training_ready=False)
    save()
except BaseException as error:
    receipt.update(status='FAILED_TMR69D_NUMERIC_DATA_ACCEPTANCE_ALL_OUTPUTS_RETAINED', error=repr(error), traceback=traceback.format_exc())
    save()
    raise
