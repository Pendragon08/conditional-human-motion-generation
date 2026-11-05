"""Extend the time-aligned caption evidence to the projected eight-case preview."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from caption_interval_alignment import associate_caption

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD = ROOT / 'bounded_joint_projection_20261031'
NEW = ROOT / 'eight_projected_preflight_20261031'
CROP = ROOT / 'mosh_000798_crop_candidate_20261031'
HUMAN = ROOT / 'mosh_000798_fullrate_20261031/output'
ARCHIVE = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old = json.loads((OLD / 'caption_association.json').read_text())
    old_audit = json.loads((OLD / 'caption_association_audit.json').read_text())
    old_manifest = OLD / 'output/manifest.json'
    if (sha(OLD / 'caption_association.json') != old_audit['report_sha256'] or
        sha(old_manifest) != old['manifest_sha256'] or sha(ARCHIVE) != old['caption_archive_sha256']):
        raise ValueError('prior associations changed')
    prior = json.loads(old_manifest.read_text())['records']
    manifest = NEW / 'manifest.json'
    rows = json.loads(manifest.read_text())['records']
    build = json.loads((NEW / 'build_report.json').read_text())
    if sha(manifest) != build['manifest_sha256'] or rows[:-1] != prior or len(rows) != 8:
        raise ValueError('unexpected eight-case manifest')
    for item, evidence in zip(prior, old['cases']):
        if item['sample_id'] != evidence['sample_id'] or item['feature_sha256'] != evidence['feature_sha256']:
            raise ValueError('prior cases differ')
    report = json.loads((CROP / 'report.json').read_text())
    human = json.loads((HUMAN / 'human_report.json').read_text())
    rec = rows[-1]
    if (rec['sample_id'] != 'MOSH_000798_crop' or rec['feature_sha256'] != report['output_sha256'] or
        sha(CROP / 'candidate.npz') != rec['feature_sha256'] or
        sha(HUMAN / 'human_120hz.npz') != human['artifact_sha256']):
        raise ValueError('new source identity changed')
    with np.load(HUMAN / 'human_120hz.npz', allow_pickle=False) as d:
        source_frames = d['source_frames'].copy()
        times = d['source_times'].copy()
    with np.load(CROP / 'candidate.npz', allow_pickle=False) as d:
        frames = d['frames_20hz'].copy()
        n = len(d['features'])
    ix = np.searchsorted(source_frames, frames)
    if not np.array_equal(source_frames[ix], frames) or len(frames) != n + 1:
        raise ValueError('source timeline mapping failed')
    clip = times[::6]
    feature = times[ix]
    with zipfile.ZipFile(ARCHIVE) as z:
        raw = z.read('texts/000798.txt')
    if hashlib.sha256(raw).hexdigest() != report['caption_member_sha256'] or report['caption_member_sha256'] != human['caption_member_sha256']:
        raise ValueError('source captions changed')
    captions = []
    for index, line in enumerate(raw.decode('utf8').splitlines(), 1):
        parts = line.split('#')
        if len(parts) != 4:
            raise ValueError('bad caption row')
        relation = associate_caption(clip_sample_times=clip, feature_sample_times=feature,
                                    start_tag=parts[2], end_tag=parts[3], timed_tag_convention='unknown')
        captions.append({'line': index, 'caption': parts[0], 'raw_tags': parts[2:], 'association': relation})
    if len(captions) != 3 or rec['caption'] != captions[0]['caption']:
        raise ValueError('caption selection mismatch')
    case = {'sample_id': rec['sample_id'], 'split': rec['split'], 'caption_clip_id': '000798',
            'caption_member_sha256': report['caption_member_sha256'], 'feature_sha256': rec['feature_sha256'],
            'time_basis': 'conditional MoSh candidate and declared 120 Hz',
            'clip_first_last_s': [float(clip[0]), float(clip[-1])],
            'feature_first_last_s': [float(feature[0]), float(feature[-1])],
            'feature_rows': n, 'captions': captions, 'caption_accepted': False}
    out = {'manifest_sha256': sha(manifest), 'prior_association_sha256': sha(OLD / 'caption_association.json'),
           'caption_archive_sha256': sha(ARCHIVE), 'cases': old['cases'] + [case], 'training_ready': False,
           'limitations': ['Association is conditional on source identity and timing, not semantic acceptance.',
                           '0/0 whole-clip descriptions do not necessarily describe the crop.']}
    with (HERE / 'caption_association.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'cases': len(out['cases']), 'captions': sum(len(x['captions']) for x in out['cases']),
                      'training_ready': False}))


if __name__ == '__main__':
    main()
