"""Independently inspect G1 joint limits before and after SFU filtering."""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE = Path(__file__).resolve().parent
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan(q, frames, names, limits):
    largest = {'excess_rad': 0., 'joint': None, 'frame': None, 'value_rad': None, 'limit_rad': None}
    count = 0
    frames_with_violation = set()
    for row_index in range(len(frames)):
        for joint_index, name in enumerate(names):
            value = float(q[row_index][joint_index])
            low, high = limits[name]
            excess = max(low - value, value - high, 0.)
            if excess > 0:
                count += 1
                frames_with_violation.add(int(frames[row_index]))
            if excess > largest['excess_rad']:
                largest = {'excess_rad': excess, 'joint': name, 'frame': int(frames[row_index]),
                           'value_rad': value, 'limit_rad': low if value < low else high}
    return {'largest': largest, 'violating_joint_values': count,
            'violating_frames': sorted(frames_with_violation), 'frames_checked': len(frames)}


def main():
    report_path = HERE / 'report.json'
    candidate = HERE / 'candidate.npz'
    report = json.loads(report_path.read_text())
    assert sha(candidate) == report['output_sha256'] and sha(URDF) == report['urdf_sha256']
    limits = {}
    for joint in ET.parse(URDF).getroot().findall('joint'):
        if joint.get('type') == 'revolute':
            node = joint.find('limit')
            limits[joint.get('name')] = (float(node.get('lower')), float(node.get('upper')))
        elif joint.get('type') == 'continuous':
            limits[joint.get('name')] = (-float('inf'), float('inf'))
    with np.load(candidate, allow_pickle=False) as data:
        names = [str(n) for n in data['joint_names']]
        raw_frames = data['raw_frames'].copy()
        blended = data['blended_120hz'][:, 7:].copy()
        frames = data['frames_20hz'].copy()
        filtered = data['filtered_20hz'][:, 7:].copy()
    assert len(names) == 27 and set(names) == set(limits)
    before = scan(blended, raw_frames, names, limits)
    after = scan(filtered, frames, names, limits)
    before_sampled = scan(blended[np.searchsorted(raw_frames, frames)], frames, names, limits)
    assert abs(before['largest']['excess_rad'] - report['max_joint_limit_violation_before_filter_rad']) < 1e-12
    assert abs(after['largest']['excess_rad'] - report['max_joint_limit_violation_after_filter_rad']) < 1e-12
    worst = after['largest']
    index = names.index(worst['joint'])
    at_raw = float(blended[np.where(raw_frames == worst['frame'])[0][0], index])
    result = {'candidate_report_sha256': sha(report_path), 'candidate_sha256': sha(candidate), 'urdf_sha256': sha(URDF),
              'before_all_120hz': before, 'before_on_20hz_source_grid': before_sampled, 'after_20hz_filter': after,
              'worst_filtered_joint_unfiltered_same_frame_rad': at_raw, 'training_ready': False,
              'limitations': ['Compares saved data only; does not independently recompute FIR, FK or solver.',
                              'Uses static URDF limits; source time and physical feasibility remain unverified.',
                              'Filter overshoot is a numeric observation, not proof of the underlying motion.']}
    with (HERE / 'joint_limit_audit.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
