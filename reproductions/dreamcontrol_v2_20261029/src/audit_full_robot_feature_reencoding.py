"""Re-encode all seven preview motions from saved robot states, without modifying them."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from feature263 import encode
from g1_fk import URDFFK

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW = ROOT / 'seven_candidate_technical_preview_20261031'
STATES = ROOT / 'state_preflight_v1_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    torch.set_num_threads(1)
    manifest_path = PREVIEW / 'experimental_manifest_not_for_training.json'
    evidence_path = STATES / 'state_manifest.json'
    cfg_path = ROOT / 'state_preflight_v3_20261031/preflight_config.json'
    cfg = json.loads(cfg_path.read_text())
    state_manifest = json.loads(evidence_path.read_text())
    if sha(manifest_path) != state_manifest['preview_manifest_sha256'] or sha(URDF) != cfg['state_urdf_sha256']:
        raise ValueError('manifest or URDF changed')
    states = {r['sample_id']: r for r in state_manifest['records']}
    fk = URDFFK(URDF)
    spans = {'root': (0, 4), 'relative_positions': (4, 67), 'rotations_6d': (67, 193),
             'local_velocities': (193, 259), 'foot_velocity_proxies': (259, 263)}
    rows = []
    for rec in json.loads(manifest_path.read_text())['records']:
        evidence = states[rec['sample_id']]
        feature = PREVIEW / rec['feature_path']
        state = STATES / evidence['state_path']
        if sha(feature) != rec['feature_sha256'] or sha(state) != evidence['state_sha256']:
            raise ValueError('input hash mismatch')
        with np.load(feature, allow_pickle=False) as f:
            expected = f['features'].copy()
        with np.load(state, allow_pickle=False) as f:
            root, q = f['root_wxyz_xyz'].copy(), f['joint_pos'].copy()
            names = [str(x) for x in f['joint_names']]
        points = fk.forward(root[:, 4:], root[:, [1, 2, 3, 0]], q, names)
        actual = encode(points, foot_velocity_squared_threshold=.002)
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('invalid re-encoding')
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        rows.append({'sample_id': rec['sample_id'], 'feature_sha256': sha(feature), 'state_sha256': sha(state),
                     'max_abs_diff': float(diff.max()),
                     'component_max_abs_diff': {key: float(diff[:, lo:hi].max()) for key, (lo, hi) in spans.items()},
                     'changed_foot_proxy_values': int(np.count_nonzero(actual[:, -4:] != expected[:, -4:])),
                     'training_ready': False})
    result = {'manifest_sha256': sha(manifest_path), 'state_manifest_sha256': sha(evidence_path),
              'urdf_sha256': sha(URDF), 'foot_velocity_squared_threshold': .002, 'cases': rows,
              'training_ready': False,
              'limitations': ['Reuses project FK, HumanML extractor and encoder; not independent original motion verification.',
                              'Foot features are a velocity proxy and the threshold is experimental.',
                              'Re-encoding reproduces saved features but does not validate source, caption or physics.']}
    with (HERE / 'report.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'case_count': len(rows), 'largest_absolute_difference': max(r['max_abs_diff'] for r in rows),
                      'changed_foot_proxy_values': sum(r['changed_foot_proxy_values'] for r in rows), 'training_ready': False}))


if __name__ == '__main__':
    main()
