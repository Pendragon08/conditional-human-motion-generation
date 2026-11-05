"""Independently verify original caption bytes and saved interval associations."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
NEW = ROOT / 'eight_projected_preflight_20261031'
ARCHIVE = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result = json.loads((HERE / 'caption_association.json').read_text())
    assert sha(ARCHIVE) == result['caption_archive_sha256']
    assert sha(NEW / 'manifest.json') == result['manifest_sha256']
    records = json.loads((NEW / 'manifest.json').read_text())['records']
    states = {x['sample_id']: x for x in json.loads((NEW / 'state_manifest.json').read_text())['records']}
    statuses = {}; total = 0; cases = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for record, case in zip(records, result['cases']):
            assert record['sample_id'] == case['sample_id'] and record['feature_sha256'] == case['feature_sha256']
            assert sha(NEW / record['feature_path']) == case['feature_sha256']
            state = states[record['sample_id']]
            assert sha(NEW / state['state_path']) == state['state_sha256']
            with np.load(NEW / state['state_path'], allow_pickle=False) as f:
                times = f['sample_times'].astype(np.float64)
            assert len(times) == case['feature_rows'] + 1
            np.testing.assert_allclose([times[0], times[-1]], case['feature_first_last_s'], rtol=0, atol=1e-8)
            raw = z.read('texts/' + case['caption_clip_id'] + '.txt')
            assert hashlib.sha256(raw).hexdigest() == case['caption_member_sha256']
            lines = raw.decode('utf8').splitlines()
            assert len(lines) == len(case['captions'])
            for line, saved in zip(lines, case['captions']):
                text, _, start, end = line.split('#')
                assert (text, [start, end]) == (saved['caption'], saved['raw_tags'])
                assoc = saved['association']
                if float(start) == 0 and float(end) == 0:
                    lo, hi = case['clip_first_last_s']
                    expected = [i for i in range(len(times)-1) if times[i] >= lo-1e-9 and times[i+1] <= hi+1e-9]
                    assert assoc['matching_feature_intervals'] == expected
                    assert assoc['tag_interpretation'] == 'whole_clip_convention'
                elif assoc['status'] != 'invalid_or_unknown_tags':
                    assert assoc['status'] == 'unknown_time_convention' and assoc['matching_feature_intervals'] == []
                assert not assoc['caption_accepted']
                statuses[assoc['status']] = statuses.get(assoc['status'], 0) + 1
                total += 1
            cases.append({'sample_id': record['sample_id'], 'captions': len(lines),
                          'head_crop_s': float(times[0] - case['clip_first_last_s'][0]),
                          'tail_crop_s': float(case['clip_first_last_s'][1] - times[-1])})
    assert len(cases) == len(records) == len(result['cases']) == 8
    out = {'association_sha256': sha(HERE / 'caption_association.json'), 'manifest_sha256': sha(NEW / 'manifest.json'),
           'caption_archive_sha256': sha(ARCHIVE), 'cases': cases, 'caption_count': total,
           'association_statuses': statuses, 'caption_semantics_accepted': 0, 'training_ready': False,
           'limitations': 'Original sample time/source identity, actual caption semantics and physical quality are not validated.'}
    with (HERE / 'independent_audit.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
