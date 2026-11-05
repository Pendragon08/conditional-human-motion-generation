"""Verify source and sheet coverage for KIT review aids; not visual acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
REVIEW = ROOT / 'kit_dart_pair_review_20261102'
HUMAN = ROOT / 'kit_dart_hypothesis_human_20261102'
ROBOT = ROOT / 'kit_dart_hypothesis_263_preview_20261102'
EXPECTED = {'001199', '007754', '006923', '001171', '007259', '001893', '009236'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    original = REVIEW / 'report.json'
    original_hash = sha(original)
    report = json.loads(original.read_text())
    if {c['id'] for c in report['cases']} != EXPECTED or len(report['cases']) != len(EXPECTED):
        raise ValueError('unexpected or repeated IDs')
    results = []
    for case in report['cases']:
        sid = case['id']
        human = HUMAN / ('human_' + sid) / 'human_20hz_dart_hypothesis.npz'
        robot = ROBOT / (sid + '.npz')
        if sha(human) != case['human_sha256'] or sha(robot) != case['robot_sha256']:
            raise ValueError('input hash mismatch')
        with np.load(human, allow_pickle=False) as h, np.load(robot, allow_pickle=False) as r:
            source, output = h['source_frames'].copy(), r['source_frames'].copy()
            if not np.array_equal(output, source[1:]):
                raise ValueError('source alignment mismatch')
        expected = 0
        for sheet in case['sheets']:
            target = REVIEW / sheet['file']
            if target.parent != REVIEW or sha(target) != sheet['sha256']:
                raise ValueError('sheet path or hash mismatch')
            if sheet['first_index'] != expected or sheet['last_index'] != min(expected + 7, len(output)-1):
                raise ValueError('sheet indices do not cover sequential frames')
            with Image.open(target) as im:
                im.verify()
            expected = sheet['last_index'] + 1
        if expected != len(output) or case['aligned_frames'] != len(output):
            raise ValueError('incomplete sheet coverage')
        results.append({'id': sid, 'aligned_frames': expected, 'sheets': len(case['sheets']),
                        'human_sha256': case['human_sha256'], 'robot_sha256': case['robot_sha256']})
    if sha(original) != original_hash:
        raise RuntimeError('render report changed during audit')
    result = {'cases': results, 'render_report_sha256': original_hash, 'script_sha256': sha(Path(__file__)),
              'caption_accepted': False, 'training_ready': False,
              'limitations': ['Sheet hashes and index coverage do not verify rendered pixels.',
                              '100 Hz is a hypothesis; semantic, identity and physical quality remain unaccepted.']}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({'cases': len(results), 'frames': sum(x['aligned_frames'] for x in results),
                      'sheets': sum(x['sheets'] for x in results), 'training_ready': False}))


if __name__ == '__main__':
    main()
