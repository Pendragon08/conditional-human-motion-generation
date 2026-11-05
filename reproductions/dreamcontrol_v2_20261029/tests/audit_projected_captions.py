"""Recheck caption archive members and interval assignment without alignment module."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE = Path(__file__).resolve().parent
PREVIEW = ROOT / 'bounded_joint_projection_20261031/output'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = HERE / 'caption_association.json'; report = json.loads(path.read_text())
    manifest_path = PREVIEW / 'manifest.json'
    assert sha(manifest_path) == report['manifest_sha256']
    archive_path = HUMANML / 'texts.zip'
    assert sha(archive_path) == report['caption_archive_sha256']
    manifest = {r['sample_id']: r for r in json.loads(manifest_path.read_text())['records']}
    assert set(manifest) == {r['sample_id'] for r in report['cases']}
    count = 0; full = 0; summarized = []
    with zipfile.ZipFile(archive_path) as archive:
        for item in report['cases']:
            rec = manifest[item['sample_id']]
            feature_path = PREVIEW / rec['feature_path']
            assert sha(feature_path) == rec['feature_sha256'] == item['feature_sha256']
            with np.load(feature_path, allow_pickle=False) as f:
                times = f['sample_times']; length = len(f['features'])
            assert len(times) == length + 1 == item['feature_rows'] + 1
            raw = archive.read('texts/' + item['caption_clip_id'] + '.txt')
            assert hashlib.sha256(raw).hexdigest() == item['caption_member_sha256']
            lines = [line.split('#') for line in raw.decode('utf8').splitlines()]
            assert len(lines) == len(item['captions'])
            for line, caption in zip(lines, item['captions']):
                assert len(line) == 4 and caption['caption'] == line[0] and caption['raw_tags'] == line[2:]
                association = caption['association']
                assert association['caption_accepted'] is False
                if line[2:] == ['0.0', '0.0']:
                    lo, hi = item['clip_first_last_s']
                    indices = [i for i in range(length) if times[i] >= lo - 1e-9 and times[i + 1] <= hi + 1e-9]
                    assert indices == association['matching_feature_intervals']
                    assert association['feature_covers_full_annotation_sample_extent'] == (abs(times[0]-lo) <= 1e-9 and abs(times[-1]-hi) <= 1e-9)
                    full += int(association['feature_covers_full_annotation_sample_extent'])
                count += 1
            assert item['caption_accepted'] is False
            summarized.append({'sample_id': item['sample_id'], 'captions': len(lines),
                               'head_crop_s': float(times[0] - item['clip_first_last_s'][0]),
                               'tail_crop_s': float(item['clip_first_last_s'][1] - times[-1])})
    result = {'report_sha256': sha(path), 'cases': summarized, 'caption_count': count,
              'full_extent_caption_count': full, 'training_ready': False,
              'limitations': ['Independently checks saved clip endpoints, not true source timing.',
                              'Cannot establish semantic meaning, source identity or physical quality.']}
    with (HERE / 'caption_association_audit.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'cases': len(summarized), 'captions': count, 'full_extent': full}))


if __name__ == '__main__': main()
