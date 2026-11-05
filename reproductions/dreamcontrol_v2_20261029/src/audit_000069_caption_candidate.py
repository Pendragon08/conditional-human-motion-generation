"""Independently recheck original labels and interval arithmetic for 000069."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from g1_fk import URDFFK

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
DATA = ROOT / 'five_conservative_preview_20261031'
VIDEO = ROOT / 'five_caption_review_videos_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
ARCHIVE = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report = json.loads((HERE / 'report.json').read_text())
    assert sha(HERE / 'intervals.jsonl') == report['intervals_sha256']
    assert sha(VIDEO / '000069.mp4') == report['video_sha256']
    source = Path(report['source_evidence']['path']); assert sha(source) == report['source_evidence']['sha256']
    with zipfile.ZipFile(ARCHIVE) as z:
        raw = z.read('texts/000069.txt')
    assert hashlib.sha256(raw).hexdigest() == report['caption_member_sha256']
    captions = [x.split('#') for x in raw.decode('utf8').splitlines()]
    assert captions[0][0] == report['chosen_in_existing_preview']['caption']
    assert captions[2][0] == report['proposed_alternative_original_line']['caption']
    state = next(x for x in json.loads((DATA / 'state_manifest.json').read_text())['records'] if x['sample_id'] == '000069')
    assert sha(DATA / state['state_path']) == report['state_sha256']
    with np.load(DATA / state['state_path'], allow_pickle=False) as f:
        roots, q, times, names = (f[x].copy() for x in ('root_wxyz_xyz','joint_pos','sample_times','joint_names'))
    with np.load(source, allow_pickle=False) as f:
        source_times = f['source_times'].copy()
        hm = f['original_joints_humanml'].copy()
    human_source = np.stack([-hm[...,0], hm[...,2], hm[...,1]], axis=-1).astype(np.float64)
    human = []
    for time in times:
        right = int(np.searchsorted(source_times, time, side='right'))
        if right == len(source_times):
            assert abs(time-source_times[-1]) < 1e-9
            human.append(human_source[-1]); continue
        left = right - 1
        assert left >= 0
        alpha = (time-source_times[left])/(source_times[right]-source_times[left])
        human.append((1-alpha)*human_source[left] + alpha*human_source[right])
    human = np.asarray(human)
    robot = URDFFK(URDF).forward(roots[:,4:], roots[:,[1,2,3,0]], q, list(names))
    rows = [json.loads(x) for x in (HERE/'intervals.jsonl').read_text().splitlines()]
    assert len(rows) == len(times)-1 == 46
    max_err = 0.
    for i, row in enumerate(rows):
        dt = float(times[i+1] - times[i]); assert row['interval'] == i
        assert abs(row['mid_s']-(times[i]+times[i+1])/2) < 1e-12
        for key, points in [('human',human),('robot',robot)]:
            for j, ankle in enumerate((7,8)):
                delta = [(points[i+1,ankle,k]-points[i+1,0,k])-(points[i,ankle,k]-points[i,0,k]) for k in range(3)]
                speed = sum(x*x for x in delta)**.5 / dt
                max_err = max(max_err, abs(float(speed)-row[key+'_ankle_speed_m_s'][j]))
            xy = sum((points[i+1,0,k]-points[i,0,k])**2 for k in range(2))**.5 / dt
            max_err = max(max_err, abs(float(xy)-row[key+'_pelvis_xy_speed_m_s']))
    assert max_err < 1e-10
    for name, (lo, hi) in report['review_periods_s'].items():
        subset = [r for r in rows if lo <= r['mid_s'] < hi]
        for kind in ('human','robot'):
            speeds = [x for r in subset for x in r[kind+'_ankle_speed_m_s']]
            saved = report['period_statistics'][name][kind]
            assert len(subset) == saved['interval_count']
            assert abs(sum(speeds)/len(speeds)-saved['ankle_relative_mean_speed_m_s']) < 1e-10
    out = {'report_sha256': sha(HERE/'report.json'), 'intervals_sha256': sha(HERE/'intervals.jsonl'),
           'intervals_verified': len(rows), 'max_scalar_speed_error': max_err, 'original_label_lines_verified': [1,3],
           'caption_accepted': False, 'training_ready': False,
           'limitations': 'Reuses robot FK; numeric motion proxies do not establish standing contact, exact speed or semantic acceptance.'}
    with (HERE/'audit_report.json').open('x',encoding='utf8') as f:
        json.dump(out,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
