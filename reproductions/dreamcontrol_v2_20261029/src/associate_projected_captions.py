"""Pair all original captions with bounded-projection preview intervals.

Frame associations are conditional on previously recorded source/time mapping.
This never confers semantic, source, physical, or training acceptance.
"""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from caption_interval_alignment import associate_caption

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
PREVIEW = ROOT / 'bounded_joint_projection_20261031/output'
OLD = ROOT / 'six_caption_time_alignment_20261031'
HERE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = PREVIEW / 'manifest.json'; projection_report = PREVIEW / 'report.json'
    manifest = json.loads(manifest_path.read_text()); projection = json.loads(projection_report.read_text())
    if sha(manifest_path) != projection['manifest_sha256']: raise ValueError('manifest changed')
    prior = json.loads((OLD / 'report.json').read_text())
    audit = json.loads((OLD / 'independent_audit.json').read_text())
    if sha(OLD / 'report.json') != audit['report_sha256']: raise ValueError('prior association changed')
    previous = {r['sample_id']: r for r in prior['cases']}
    view = json.loads((ROOT / 'source_grouping_v3_20261031/six_source_preview.json').read_text())
    index = {r['sample_id']: r for r in view['records']}
    human_path = ROOT / 'accad_004347_fullrate_20261031/human_report.json'
    human = json.loads(human_path.read_text())
    human_audit = json.loads((human_path.parent / 'human_audit.json').read_text())
    if sha(human_path) != human_audit['human_report_sha256'] or human['fps_declared'] != 120:
        raise ValueError('ACCAD source evidence changed')
    rows = []
    archive_path = HUMANML / 'texts.zip'
    with zipfile.ZipFile(archive_path) as archive:
        for rec in manifest['records']:
            sample = rec['sample_id']; feature_file = PREVIEW / rec['feature_path']
            if sha(feature_file) != rec['feature_sha256']: raise ValueError('feature changed')
            with np.load(feature_file, allow_pickle=False) as data:
                times = data['sample_times'].copy(); n = len(data['features'])
            if len(times) != n + 1: raise ValueError('feature endpoints missing')
            if sample.startswith('BML_'):
                annotation = index[sample]['candidate_clip_annotations'][0]
                clipid = annotation['clip_id']; start, stop = annotation['index_slice']
                clip = np.arange(start, stop, dtype=float) / 20.
                basis = 'conditional BML candidate and declared 120 Hz'
            elif sample == 'ACCAD_004347':
                clipid = '004347'
                clip = np.arange(0, human['frames'], 6, dtype=float) / human['fps_declared']
                basis = 'conditional ACCAD candidate and declared 120 Hz'
            else:
                clipid = sample
                folder = (ROOT / 'humanml_mirror_time_witness_v2_20261030' if sample == '000008'
                          else ROOT / 'ssm_three_human_candidates_20261031')
                with np.load(folder / (sample + '_joint_witness.npz'), allow_pickle=False) as d:
                    clip = d['source_times'].copy()
                basis = 'reference-derived SSM timing, conditional source identity'
            raw = archive.read('texts/' + clipid + '.txt'); member_sha = hashlib.sha256(raw).hexdigest()
            if sample in previous and member_sha != previous[sample]['caption_member_sha256']:
                raise ValueError('caption member differs from prior audit')
            if sample == 'ACCAD_004347' and member_sha != human['caption_member_sha256']:
                raise ValueError('ACCAD caption member changed')
            captions = []
            for line_no, line in enumerate(raw.decode('utf8').splitlines(), 1):
                fields = line.split('#')
                if len(fields) != 4: raise ValueError('invalid HumanML caption row')
                try:
                    association = associate_caption(clip_sample_times=clip, feature_sample_times=times,
                        start_tag=fields[2], end_tag=fields[3], timed_tag_convention='unknown')
                except ValueError as exc:
                    association = {'status': 'invalid_or_unknown_tags', 'reason': str(exc),
                                   'caption_accepted': False, 'matching_feature_intervals': []}
                captions.append({'line': line_no, 'caption': fields[0], 'raw_tags': fields[2:],
                                 'association': association})
            chosen = [x for x in captions if x['caption'] == rec['caption']]
            if len(chosen) != 1: raise ValueError('chosen caption ambiguous')
            if sample in previous:
                old = previous[sample]['association']; current = chosen[0]['association']
                if old['matching_feature_intervals'] != current['matching_feature_intervals']:
                    raise ValueError('projection changed caption interval assignment')
            rows.append({'sample_id': sample, 'split': rec['split'], 'caption_clip_id': clipid,
                         'caption_member_sha256': member_sha, 'feature_sha256': rec['feature_sha256'],
                         'time_basis': basis, 'clip_first_last_s': [float(clip[0]), float(clip[-1])],
                         'feature_first_last_s': [float(times[0]), float(times[-1])],
                         'feature_rows': n, 'captions': captions, 'caption_accepted': False})
    result = {'manifest_sha256': sha(manifest_path), 'projection_report_sha256': sha(projection_report),
              'prior_six_report_sha256': sha(OLD / 'report.json'), 'caption_archive_sha256': sha(archive_path),
              'cases': rows, 'training_ready': False,
              'limitations': ['0/0 is a whole-clip convention, not semantic validation.',
                              'Timing and source identity remain conditional; no original 3D identity proof.',
                              'Per-caption interval association does not verify that a cropped clip preserves the action.']}
    with (HERE / 'caption_association.json').open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'clips': len(rows), 'captions': sum(len(r['captions']) for r in rows),
                      'training_ready': False}))


if __name__ == '__main__': main()
