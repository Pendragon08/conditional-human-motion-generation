"""Independently check exported frame correspondence and canonical positions.

Does not independently recompute the robot FK or establish physical quality.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.report.read_text(encoding='utf-8'))
    assert report['training_ready'] is False
    assert sha(a.artifact) == report['input_sha256']
    assert sha(a.features) == report['output_sha256']
    with np.load(a.artifact, allow_pickle=False) as src, np.load(a.features, allow_pickle=False) as out:
        frames = src['frames20']
        np.testing.assert_array_equal(frames, out['source_frames'])
        assert frames.tolist() == report['source_frames']
        assert np.all(np.diff(frames) == 6)
        features = out['features']
        points = out['points_z_up_world']
        decoded = out['decoded_points']
    assert features.shape == (len(frames) - 1, 263)
    assert points.shape == (len(frames), 22, 3)
    assert decoded.shape == (len(frames) - 1, 22, 3)
    assert np.isfinite(features).all() and np.isfinite(points).all()
    assert np.isin(features[:, -4:], [0., 1.]).all()
    transformed = np.stack((points[:, :, 0], points[:, :, 2], -points[:, :, 1]), axis=-1)
    transformed[:, :, 1] -= transformed[:, :, 1].min()
    transformed[:, :, 0] -= transformed[0, 0, 0]
    transformed[:, :, 2] -= transformed[0, 0, 2]
    across = transformed[0, 2] - transformed[0, 1] + transformed[0, 17] - transformed[0, 16]
    angle = -math.atan2(float(across[2]), -float(across[0]))
    c, s = math.cos(angle), math.sin(angle)
    canonical = transformed.copy()
    canonical[:, :, 0] = c * transformed[:, :, 0] + s * transformed[:, :, 2]
    canonical[:, :, 2] = -s * transformed[:, :, 0] + c * transformed[:, :, 2]
    max_error = float(np.linalg.norm(canonical[:-1] - decoded, axis=-1).max())
    assert max_error < 1e-4
    assert math.isclose(max_error, report['max_reconstruction_error_m'], rel_tol=1e-6, abs_tol=1e-12)
    result = {'artifact_sha256': sha(a.artifact), 'features_sha256': sha(a.features),
              'report_sha256': sha(a.report), 'frames_verified': len(frames),
              'feature_rows_verified': len(features), 'canonical_error_m': max_error,
              'training_ready': False,
              'limitation': 'Independent frame/hash/canonical check; FK, source FPS, physics and training acceptance not established.'}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
