"""Independently check the six-preview trajectory diagnostic, with scalar loops."""
import bisect
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW = ROOT / 'six_candidate_technical_preview_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def interpolate(q, times, grid):
    out = []
    for stamp in grid:
        idx = bisect.bisect_left(times, stamp)
        if idx == 0:
            out.append(q[0])
        elif idx == len(times):
            out.append(q[-1])
        else:
            alpha = (stamp - times[idx - 1]) / (times[idx] - times[idx - 1])
            out.append([float(a) + alpha * (float(b) - float(a)) for a, b in zip(q[idx - 1], q[idx])])
    return out


def verify(q, times, names, limits, record):
    excess, speed, acceleration = 0., 0., 0.
    for j, name in enumerate(names):
        low, high = limits[name]
        vals = [float(row[j]) for row in q]
        for val in vals:
            excess = max(excess, low - val, val - high)
        v = [(vals[i + 1] - vals[i]) / (times[i + 1] - times[i]) for i in range(len(vals) - 1)]
        for value in v:
            speed = max(speed, abs(value))
        for i in range(len(v) - 1):
            acceleration = max(acceleration, abs((v[i + 1] - v[i]) / ((times[i + 2] - times[i]) / 2)))
    for key, actual in [('max_limit_excess_rad', excess), ('max_joint_speed_rad_s', speed), ('max_joint_acceleration_rad_s2', acceleration)]:
        if not math.isclose(actual, record[key], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError((key, actual, record[key]))
    if record['samples'] != len(times):
        raise ValueError('sample count')


def main():
    report_path = HERE / 'report.json'
    report = json.loads(report_path.read_text())
    manifest_path = PREVIEW / 'experimental_manifest_not_for_training.json'
    if digest(manifest_path) != report['manifest_sha256'] or digest(URDF) != report['urdf_sha256']:
        raise ValueError('manifest or URDF mismatch')
    manifest = json.loads(manifest_path.read_text())
    joints = ET.parse(URDF).getroot().findall('joint')
    limits = {j.attrib['name']: (float(j.find('limit').attrib['lower']), float(j.find('limit').attrib['upper'])) for j in joints if j.attrib['type'] == 'revolute'}
    by_id = {rec['sample_id']: rec for rec in manifest['records']}
    for row in report['cases']:
        mid = row['sample_id']
        folder = (ROOT / 'pyroki_ssm_000008_global_20261031' if mid == '000008' else
                  ROOT / 'bml_joint_smoothness_20261031' / mid[4:] if mid.startswith('BML_') else
                  ROOT / 'ssm_three_global_20261031' / mid)
        trajectory = folder / 'trajectory.npz'
        feature = PREVIEW / by_id[mid]['feature_path']
        if digest(trajectory) != row['trajectory_sha256'] or digest(feature) != row['feature_sha256']:
            raise ValueError('artifact mismatch')
        with np.load(trajectory, allow_pickle=False) as data:
            q = data['joint_pos'].tolist()
            times = data['source_times'].tolist()
            names = [str(v) for v in data['joint_names']]
        with np.load(feature, allow_pickle=False) as data:
            grid = data['source_times' if mid.startswith('BML_') else 'time_grid'].tolist()
        if len(set(names)) != 27 or set(names) != set(limits) or grid[0] < times[0] - 1e-10 or grid[-1] > times[-1] + 1e-10:
            raise ValueError('mapping or grid')
        verify(q, times, names, limits, row['raw_trajectory'])
        verify(interpolate(q, times, grid), grid, names, limits, row['feature_grid_linear_joints'])
    result = {'input_report_sha256': digest(report_path), 'manifest_sha256': digest(manifest_path),
              'urdf_sha256': digest(URDF), 'case_count': len(report['cases']), 'scalar_checks_passed': True,
              'limitations': ['Shares original artifacts and URDF; does not verify timestamps or physics.',
                              'Independently verifies scalar maxima and sample counts, not extrema locations or original feature encoding.']}
    with (HERE / 'independent_audit.json').open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
