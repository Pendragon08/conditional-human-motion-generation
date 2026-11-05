"""Recompute all eight preview joint-rate diagnostics, with a scalar cross-check.

20 Hz interval averages and URDF declarations are not a physical acceptance or
an author filter. This reads existing assets and writes only a new report.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from audit_declared_joint_velocity import audit_velocity, read_velocity_limits

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW = ROOT / 'eight_ssm_caption_preview_20261101'
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    config_path = PREVIEW / 'config.json'
    config_sha = sha(config_path)
    cfg = json.loads(config_path.read_text())
    manifest = Path(cfg['state_manifest'])
    manifest_sha = sha(manifest)
    urdf = Path(cfg['state_urdf'])
    urdf_sha = sha(urdf)
    if urdf_sha != cfg['state_urdf_sha256']:
        raise ValueError('URDF pin mismatch')
    limits = read_velocity_limits(urdf)
    scalar_limits = {}
    for joint in ET.parse(urdf).getroot().findall('joint'):
        if joint.get('type') == 'revolute':
            el = joint.find('limit')
            scalar_limits[joint.get('name')] = None if el is None or el.get('velocity') is None else float(el.get('velocity'))
    if limits != scalar_limits:
        raise ValueError('URDF parsing disagreement')
    records = json.loads(manifest.read_text())['records']
    rows = []
    for row in records:
        state = (Path(cfg['state_root']) / row['state_path']).resolve(strict=True)
        if not state.is_relative_to(Path(cfg['state_root']).resolve(strict=True)) or sha(state) != row['state_sha256']:
            raise ValueError('state pin mismatch')
        with np.load(state, allow_pickle=False) as data:
            q = np.asarray(data['joint_pos'], dtype=np.float64)
            t = np.asarray(data['sample_times'], dtype=np.float64)
            names = [str(x) for x in data['joint_names']]
        result = audit_velocity(q, t, names, limits)
        if result['status'] != 'nominal_average_rates_checked':
            rows.append({'sample_id': row['sample_id'], 'result': result, 'scalar_check': 'missing_limits'})
            continue
        greatest = -1.0
        worst = None
        count = 0
        intervals = 0
        for i in range(len(t) - 1):
            exceeded = False
            for j, name in enumerate(names):
                ratio = abs(float(q[i + 1, j]) - float(q[i, j])) / (float(t[i + 1]) - float(t[i])) / scalar_limits[name]
                if ratio > greatest:
                    greatest, worst = ratio, (i, name)
                if ratio > 1:
                    count += 1
                    exceeded = True
            intervals += int(exceeded)
        if (abs(greatest - result['maximum_ratio']) > 1e-12 or
                worst != (result['worst_interval'], result['worst_joint']) or
                count != result['joint_interval_pairs_above_limit'] or
                intervals != result['intervals_with_any_joint_above_limit']):
            raise ValueError('scalar and vector checks differ')
        if sha(state) != row['state_sha256']:
            raise RuntimeError('state changed during check')
        rows.append({'sample_id': row['sample_id'], 'state_sha256': row['state_sha256'],
                     'result': result, 'scalar_check': 'passed'})
    if sha(config_path) != config_sha or sha(manifest) != manifest_sha or sha(urdf) != urdf_sha:
        raise RuntimeError('input changed during check')
    output = {'config_sha256': config_sha, 'state_manifest_sha256': manifest_sha,
              'urdf_sha256': urdf_sha, 'script_sha256': sha(Path(__file__)),
              'module_sha256': sha(Path(__file__).parent / 'audit_declared_joint_velocity.py'),
              'cases': rows, 'training_ready': False,
              'limitations': ['Nominal time and URDF declarations only; not measured capture timing or motor limits.',
                              'Interval averages do not bound intersample peaks.',
                              'Not an author filtering rule or physical, source, license, caption or training acceptance.']}
    with (HERE / 'report.json').open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'count': len(rows), 'above_limit': sum(r['result'].get('intervals_with_any_joint_above_limit', 0) > 0 for r in rows),
                      'max_ratio': max(r['result'].get('maximum_ratio', 0) for r in rows), 'training_ready': False}))


if __name__ == '__main__':
    main()
