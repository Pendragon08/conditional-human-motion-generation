"""Propose an original 000069 caption without the unsupported 2x-speed claim.

Reports all-frame, nominal-time joint movement as review evidence; it does not
automatically accept a caption or change the training dataset.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from g1_fk import URDFFK
from render_five_caption_review import human_data

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
DATA = ROOT / 'five_conservative_preview_20261031'
VIDEO = ROOT / 'five_caption_review_videos_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def intervals(points, times):
    relative = points[:, [7, 8]] - points[:, [0]]
    displacement = np.linalg.norm(np.diff(relative, axis=0), axis=2)
    speed = displacement / np.diff(times)[:, None]
    pelvis_speed = np.linalg.norm(np.diff(points[:, 0, :2], axis=0), axis=1) / np.diff(times)
    return speed, pelvis_speed


def summary(speed, pelvis, mid, lo, hi):
    mask = (mid >= lo) & (mid < hi)
    if not mask.any():
        raise ValueError('empty review period')
    return {'interval_count': int(mask.sum()), 'ankle_relative_mean_speed_m_s': float(speed[mask].mean()),
            'ankle_relative_max_speed_m_s': float(speed[mask].max()),
            'pelvis_xy_mean_speed_m_s': float(pelvis[mask].mean()),
            'pelvis_xy_max_speed_m_s': float(pelvis[mask].max())}


def main():
    report = json.loads((VIDEO / 'report.json').read_text())
    verified = json.loads((VIDEO / 'audit_report.json').read_text())
    if sha(VIDEO / 'report.json') != verified['report_sha256'] or sha(DATA / 'manifest.json') != report['manifest_sha256']:
        raise ValueError('video or manifest changed')
    item = next(x for x in report['cases'] if x['sample_id'] == '000069')
    if sha(VIDEO / '000069.mp4') != item['video_sha256']:
        raise ValueError('review video changed')
    records = json.loads((DATA / 'manifest.json').read_text())['records']
    states = json.loads((DATA / 'state_manifest.json').read_text())['records']
    rec = next(x for x in records if x['sample_id'] == '000069')
    state = next(x for x in states if x['sample_id'] == '000069')
    if sha(DATA / state['state_path']) != item['state_sha256'] or rec['feature_sha256'] != item['feature_sha256']:
        raise ValueError('state identity changed')
    with np.load(DATA / state['state_path'], allow_pickle=False) as f:
        roots, q, times, names = (f[x].copy() for x in ('root_wxyz_xyz','joint_pos','sample_times','joint_names'))
    human, evidence = human_data('000069', times)
    robot = URDFFK(URDF).forward(roots[:, 4:], roots[:, [1,2,3,0]], q, list(names))
    if hashlib.sha256(np.ascontiguousarray(human, dtype='<f8').tobytes()).hexdigest() != item['human_aligned_points_sha256']:
        raise ValueError('human point evidence changed')
    if hashlib.sha256(np.ascontiguousarray(robot, dtype='<f8').tobytes()).hexdigest() != item['robot_points_sha256']:
        raise ValueError('robot point evidence changed')
    hs, hp = intervals(human, times); rs, rp = intervals(robot, times); mid = (times[:-1] + times[1:]) / 2
    with (HERE / 'intervals.jsonl').open('x', encoding='utf8') as f:
        for i in range(len(mid)):
            f.write(json.dumps({'interval': i, 'mid_s': float(mid[i]), 'human_ankle_speed_m_s': hs[i].tolist(),
                                'robot_ankle_speed_m_s': rs[i].tolist(), 'human_pelvis_xy_speed_m_s': float(hp[i]),
                                'robot_pelvis_xy_speed_m_s': float(rp[i])}, allow_nan=False) + '\n')
    captions = item['caption_records']
    if len(captions) != 3 or '2x speed' not in captions[0]['caption']:
        raise ValueError('caption content changed')
    proposed = captions[2]
    if proposed['caption'] != 'a person runs in place, then stands still' or proposed['raw_tags'] != ['0.0','0.0']:
        raise ValueError('original candidate caption changed')
    periods = {'moving': [.6, 1.6], 'tail': [1.85, float(times[-1]) + 1e-9]}
    stats = {}
    for key, (lo, hi) in periods.items():
        stats[key] = {'human': summary(hs, hp, mid, lo, hi), 'robot': summary(rs, rp, mid, lo, hi)}
    out = {'sample_id': '000069', 'split': rec['split'], 'feature_sha256': rec['feature_sha256'],
           'state_sha256': item['state_sha256'], 'video_sha256': item['video_sha256'],
           'caption_member_sha256': item['caption_member_sha256'], 'source_evidence': evidence,
           'chosen_in_existing_preview': captions[0], 'proposed_alternative_original_line': proposed,
           'proposed_line_number': 3, 'times_nominal_s': [float(times[0]), float(times[-1])],
           'review_periods_s': periods, 'period_statistics': stats, 'intervals_sha256': sha(HERE / 'intervals.jsonl'),
           'script_sha256': sha(Path(__file__)), 'caption_accepted': False, 'training_ready': False,
           'limitations': ['Periods chosen after visually reviewing all 47 video frames; no independent speed baseline.',
                           'Ankle motion and reduced tail speed are proxies, not proof of running or true standing contact.',
                           'Original label line 3 is only a review proposal; no manifest is changed.',
                           'Source FPS/model identity and physical quality remain unaccepted.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'sample': '000069', 'proposed_line': 3, 'caption_accepted': False, 'training_ready': False}))


if __name__ == '__main__':
    main()
