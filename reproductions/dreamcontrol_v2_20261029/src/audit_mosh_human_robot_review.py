"""Recompute 149-frame comparison metadata; this does not assess semantics."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
CROP = ROOT / 'mosh_000798_crop_candidate_20261031'
HUMAN = ROOT / 'mosh_000798_fullrate_20261031/output'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scalar_metrics(data):
    counts = [0, 0]
    high = [-float('inf'), -float('inf')]
    mins = [float('inf')] * 3
    maxs = [-float('inf')] * 3
    for frame in data:
        for j, wrist in enumerate((20, 21)):
            delta = float(frame[wrist, 2]) - float(frame[15, 2])
            counts[j] += int(delta > 0)
            high[j] = max(high[j], delta)
        for axis in range(3):
            mins[axis] = min(mins[axis], float(frame[0, axis]))
            maxs[axis] = max(maxs[axis], float(frame[0, axis]))
    return counts, high, [maxs[i] - mins[i] for i in range(3)]


def main():
    report = json.loads((HERE / 'report.json').read_text())
    movie = HERE / 'human_robot_review.mp4'
    assert sha(movie) == report['video_sha256']
    assert sha(HUMAN / 'human_120hz.npz') == report['human_sha256']
    assert sha(CROP / 'candidate.npz') == report['candidate_sha256']
    with np.load(HUMAN / 'human_120hz.npz', allow_pickle=False) as f:
        h = f['human'].copy()
        source_frames = f['source_frames'].copy()
    with np.load(CROP / 'candidate.npz', allow_pickle=False) as f:
        r, frames = f['points_z_up_world'].copy(), f['frames_20hz'].copy()
    hlist = []
    for frame in frames:
        matches = np.flatnonzero(source_frames == frame)
        assert len(matches) == 1
        hlist.append(h[matches[0]])
    h = np.asarray(hlist)
    for points, key in ((h, 'source_human_observations'), (r, 'robot_observations')):
        counts, high, spread = scalar_metrics(points)
        saved = report[key]
        assert counts == saved['left_right_wrist_above_head_frames']
        np.testing.assert_allclose(high, saved['max_wrist_minus_head_m'], atol=1e-7, rtol=0)
        np.testing.assert_allclose(spread[:2], saved['root_xy_range_m'], atol=1e-7, rtol=0)
        np.testing.assert_allclose(spread[2], saved['root_z_range_m'], atol=1e-7, rtol=0)
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,r_frame_rate,duration', '-of', 'json', str(movie)], text=True))
    stream = info['streams'][0]
    assert stream['nb_read_frames'] == '149' and stream['r_frame_rate'] == '20/1'
    out = {'report_sha256': sha(HERE / 'report.json'), 'video_sha256': sha(movie), 'verified_frames': 149,
           'source_frame_interval': [int(frames[0]), int(frames[-1])], 'encoded_duration_seconds': float(stream['duration']),
           'sample_time_span_seconds': float((frames[-1] - frames[0]) / 120),
           'metrics_recomputed': True, 'caption_accepted': False, 'training_ready': False,
           'limitations': 'No independent rendering, SMPL-X/FK rerun, actual FPS or semantic/physical acceptance.'}
    with (HERE / 'audit_report.json').open('x', encoding='utf8') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
