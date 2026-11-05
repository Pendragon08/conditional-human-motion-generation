"""Stream-check whether official variants are rigid transforms of originals.

This is a storage/processing feasibility diagnostic, not text, source rights,
physics, fitting equivalence, or training acceptance. Reads ZIP members only.
"""
import collections
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ARCHIVE = ROOT / 'data/OmniRetarget_Dataset/robot-object.zip'
PAIRS = ROOT / 'omomo_text_pairing_20261101/candidate_pairs.jsonl'
HERE = Path(__file__).resolve().parent


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def compare(original, variant):
    a, b = np.asarray(original, dtype=np.float64), np.asarray(variant, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 43 or len(a) < 2:
        return {'status': 'shape_mismatch', 'original_shape': list(a.shape), 'variant_shape': list(b.shape)}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'status': 'nonfinite'}
    def rotations(q):
        if np.any(np.linalg.norm(q, axis=1) < 1e-8):
            raise ValueError('zero quaternion')
        return Rotation.from_quat(q[:, [1, 2, 3, 0]])
    base_a, base_b = rotations(a[:, :4]), rotations(b[:, :4])
    obj_a, obj_b = rotations(a[:, 36:40]), rotations(b[:, 36:40])
    transform = base_b[0] * base_a[0].inv()
    offset = b[0, 4:7] - transform.apply(a[0, 4:7])
    base_xyz_error = np.linalg.norm(transform.apply(a[:, 4:7]) + offset - b[:, 4:7], axis=1)
    obj_xyz_error = np.linalg.norm(transform.apply(a[:, 40:43]) + offset - b[:, 40:43], axis=1)
    base_rot_error = (base_b.inv() * (transform * base_a)).magnitude()
    obj_rot_error = (obj_b.inv() * (transform * obj_a)).magnitude()
    up_error = float(np.linalg.norm(transform.apply([0., 0., 1.]) - [0., 0., 1.]))
    metrics = {'joint_max_abs_rad': float(np.max(np.abs(a[:, 7:36] - b[:, 7:36]))),
               'base_position_max_m': float(base_xyz_error.max()), 'object_position_max_m': float(obj_xyz_error.max()),
               'base_rotation_max_rad': float(base_rot_error.max()), 'object_rotation_max_rad': float(obj_rot_error.max()),
               'up_axis_error': up_error, 'vertical_translation_m': float(offset[2])}
    # Numeric tolerance for this particular representation, not a data quality
    # criterion; per-frame metrics are retained for independent review.
    close = max(abs(value) for value in metrics.values()) <= 1e-5
    return {'status': 'planar_rigid_within_numeric_tolerance' if close else 'not_planar_rigid_within_numeric_tolerance',
            'metrics': metrics, 'rotation_xyzw': transform.as_quat().tolist(), 'translation_xyz': offset.tolist()}


def main():
    begin = time.perf_counter()
    archive_hash, pair_hash = sha_file(ARCHIVE), sha_file(PAIRS)
    if archive_hash != '93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3':
        raise ValueError('archive changed')
    rows = [json.loads(line) for line in PAIRS.read_text().splitlines()]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['nominal_action_key']].append(row)
    if len(groups) != 312 or len(rows) != 1826:
        raise ValueError('unexpected source coverage')
    counts = collections.Counter()
    count_variant = collections.Counter()
    with zipfile.ZipFile(ARCHIVE) as archive, (HERE/'records.jsonl').open('x', encoding='utf8') as stream:
        for key, members in sorted(groups.items()):
            originals = [row for row in members if row['variant'] == 'original']
            if len(originals) != 1:
                raise ValueError('missing or duplicate original: ' + key)
            orig = originals[0]
            raw = archive.read(orig['motion_member'])
            if hashlib.sha256(raw).hexdigest() != orig['motion_member_sha256']:
                raise ValueError('original member hash mismatch')
            with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                oq, fps = f['qpos'].copy(), float(f['fps'])
            for variant in members:
                if variant['variant'] == 'original':
                    continue
                data = archive.read(variant['motion_member'])
                if hashlib.sha256(data).hexdigest() != variant['motion_member_sha256']:
                    raise ValueError('variant member hash mismatch')
                with np.load(io.BytesIO(data), allow_pickle=False) as f:
                    vq, vfps = f['qpos'].copy(), float(f['fps'])
                result = compare(oq, vq) if vfps == fps else {'status': 'fps_mismatch', 'original_fps': fps, 'variant_fps': vfps}
                record = {'nominal_action_key': key, 'original_member': orig['motion_member'],
                          'original_sha256': orig['motion_member_sha256'], 'variant_member': variant['motion_member'],
                          'variant_sha256': variant['motion_member_sha256'], 'variant_type': variant['variant'],
                          'comparison': result, 'training_ready': False}
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                counts[result['status']] += 1
                count_variant[variant['variant'].split('_')[0]] += 1
    if sum(counts.values()) != 1514 or sha_file(ARCHIVE) != archive_hash or sha_file(PAIRS) != pair_hash:
        raise RuntimeError('coverage mismatch or source changed')
    report = {'archive_sha256': archive_hash, 'pair_manifest_sha256': pair_hash, 'records_sha256': sha_file(HERE/'records.jsonl'),
              'script_sha256': sha_file(Path(__file__)), 'source_groups': len(groups), 'variants': sum(counts.values()),
              'result_counts': dict(counts), 'variant_kind_counts': dict(count_variant),
              'elapsed_seconds': time.perf_counter() - begin, 'training_ready': False,
              'limitations': ['Raw joint and rigid-body transforms only; no retargeter equivariance or 263D equality established.',
                              'Only filename-based grouping, not proof of identity, semantics or license.',
                              'Numeric tolerance is a representation diagnostic, not a training quality threshold.']}
    with (HERE/'report.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
