"""Classify prior full-trajectory MuJoCo self-contacts by involved bodies."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MEASURE = ROOT / 'eight_static_robot_quality_20261031'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    original = json.loads((MEASURE / 'report.json').read_text())
    audit = json.loads((MEASURE / 'audit_report.json').read_text())
    path = MEASURE / 'frames.jsonl'
    if sha(MEASURE / 'report.json') != audit['report_sha256'] or sha(path) != audit['frames_sha256']:
        raise ValueError('audited diagnostic changed')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    result = []
    for case in original['cases']:
        sample = [r for r in rows if r['sample_id'] == case['sample_id']]
        if len(sample) != case['frames']:
            raise ValueError('frame coverage changed')
        counts = Counter(); pairs = Counter(); worst = {'hand': None, 'other': None}
        for row in sample:
            flags = set()
            for contact in row['self_penetration_contacts']:
                names = contact['bodies']
                category = 'hand' if any(name.startswith(('left_hand_', 'right_hand_')) for name in names) else 'other'
                counts[category + '_contacts'] += 1
                flags.add(category)
                pairs[(category, *sorted(names))] += 1
                if worst[category] is None or contact['dist_m'] < worst[category]['dist_m']:
                    worst[category] = {'sample_index': row['sample_index'], 'bodies': names, 'dist_m': contact['dist_m']}
            for name in flags:
                counts[name + '_frames'] += 1
        result.append({'sample_id': case['sample_id'], 'frames': len(sample),
                       'any_self_contact_frames': case['frames_with_self_penetration_contact'],
                       'hand_contact_frames': counts['hand_frames'], 'other_contact_frames': counts['other_frames'],
                       'hand_contact_count': counts['hand_contacts'], 'other_contact_count': counts['other_contacts'],
                       'worst_hand_contact': worst['hand'], 'worst_other_contact': worst['other'],
                       'pairs': [{'category': key[0], 'bodies': list(key[1:]), 'contacts': count}
                                 for key, count in sorted(pairs.items(), key=lambda x: (-x[1], x[0]))]})
    out = {'diagnostic_report_sha256': sha(MEASURE / 'report.json'), 'frames_sha256': sha(path),
           'script_sha256': sha(Path(__file__)), 'cases': result, 'training_ready': False,
           'limits': ['Classification of recorded MuJoCo contacts, not collision resolution or an independent engine.',
                      'Hand geometry uses fixed default pose and differs from robot 27-DoF and Isaac assets.',
                      'No contact force, ground calibration, source or training acceptance.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'cases': len(result), 'training_ready': False}))


if __name__ == '__main__':
    main()
