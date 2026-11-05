"""Independently recount all body-classified self-contact observations."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MEASURE = ROOT / 'eight_static_robot_quality_20261031'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / 'report.json').read_text())
    if sha(MEASURE / 'frames.jsonl') != report['frames_sha256']:
        raise ValueError('frame evidence changed')
    cases = {c['sample_id']: c for c in report['cases']}
    summary = {k: {'frames': 0, 'hand_frames': 0, 'other_frames': 0, 'hand_contacts': 0, 'other_contacts': 0} for k in cases}
    for line in (MEASURE / 'frames.jsonl').read_text().splitlines():
        row = json.loads(line); item = summary[row['sample_id']]; item['frames'] += 1
        hand, other = 0, 0
        for contact in row['self_penetration_contacts']:
            if any(x.startswith('left_hand_') or x.startswith('right_hand_') for x in contact['bodies']):
                hand += 1
            else:
                other += 1
        item['hand_contacts'] += hand; item['other_contacts'] += other
        item['hand_frames'] += bool(hand); item['other_frames'] += bool(other)
    for ident, item in summary.items():
        saved = cases[ident]
        for name in item:
            if item[name] != saved[name if name == 'frames' else name.replace('_contacts', '_contact_count').replace('_frames', '_contact_frames')]:
                raise ValueError('count mismatch: ' + ident + ' ' + name)
    result = {'report_sha256': sha(HERE / 'report.json'), 'frames_sha256': sha(MEASURE / 'frames.jsonl'),
              'cases': len(cases), 'frames_recounted': sum(x['frames'] for x in summary.values()),
              'hand_contact_frames': sum(x['hand_frames'] for x in summary.values()),
              'non_hand_contact_frames': sum(x['other_frames'] for x in summary.values()),
              'training_ready': False,
              'limitations': 'Reclassifies saved results; not an independent collision engine or physical acceptance.'}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
