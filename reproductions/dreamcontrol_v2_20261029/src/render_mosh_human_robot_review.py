"""Render the whole nominal 20 Hz crop alongside its source human fit.

Human samples are direct decimation; the robot used approximate FIR. World
translation remains visible, and neither represents independently verified
human truth or a physical simulation.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
CROP = ROOT / 'mosh_000798_crop_candidate_20261031'
HUMAN = ROOT / 'mosh_000798_fullrate_20261031/output'
HERE = Path(__file__).resolve().parent
CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15),
          (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    r = json.loads((CROP / 'report.json').read_text())
    hr = json.loads((HUMAN / 'human_report.json').read_text())
    if sha(CROP / 'candidate.npz') != r['output_sha256'] or sha(HUMAN / 'human_120hz.npz') != hr['artifact_sha256']:
        raise ValueError('source or candidate changed')
    with np.load(CROP / 'candidate.npz', allow_pickle=False) as data:
        robot, frames = data['points_z_up_world'].copy(), data['frames_20hz'].copy()
    with np.load(HUMAN / 'human_120hz.npz', allow_pickle=False) as data:
        source_frames = data['source_frames'].copy()
        source = data['human'].copy()
    indices = np.searchsorted(source_frames, frames)
    if not np.array_equal(source_frames[indices], frames) or not np.array_equal(frames, np.arange(126, 1015, 6)):
        raise ValueError('frame correspondence changed')
    human = source[indices]
    if human.shape != robot.shape or human.shape != (149, 22, 3) or not np.isfinite(human).all() or not np.isfinite(robot).all():
        raise ValueError('nonfinite or invalid points')
    positions = np.concatenate([human.reshape(-1, 3), robot.reshape(-1, 3)])
    mins, maxs = positions.min(0), positions.max(0)
    center = (mins + maxs) / 2
    radius = float(max((maxs - mins).max() * .58, .4))
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    labels = ((0, 2, 'front x/z'), (1, 2, 'side y/z'))
    lines = []
    for row, title in enumerate(('Source human fit', 'Retargeted G1')):
        per_row = []
        for col, (x, y, label) in enumerate(labels):
            ax = axes[row, col]
            ax.set(xlim=(center[x] - radius, center[x] + radius), ylim=(center[y] - radius, center[y] + radius), aspect='equal')
            ax.set_title(f'{title}: {label}')
            ax.set_xlabel('world ' + 'xyz'[x] + ' (m)'); ax.set_ylabel('world ' + 'xyz'[y] + ' (m)')
            ax.grid(alpha=.2)
            per_row.append([ax.plot([], [], '-o', markersize=2, linewidth=2)[0] for chain in CHAINS])
        lines.append(per_row)
    caption = ' / '.join(x['caption'] for x in r['original_caption_records'])
    fig.suptitle('000798: retained crop; translations preserved\n' + caption, fontsize=8, wrap=True)
    timestamp = fig.text(.5, .01, '', ha='center')
    movie = HERE / 'human_robot_review.mp4'
    if movie.exists():
        raise FileExistsError(movie)
    active = FFMpegWriter(fps=20, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    with active.saving(fig, str(movie), dpi=100):
        for i, frame in enumerate(frames):
            for row, data in enumerate((human, robot)):
                for col, (x, y, _) in enumerate(labels):
                    for line, chain in zip(lines[row][col], CHAINS):
                        line.set_data(data[i, list(chain), x], data[i, list(chain), y])
            timestamp.set_text(f'Nominal source time: {frame / 120:.2f} s   |   frame {i+1}/149')
            active.grab_frame()
    plt.close(fig)
    def metrics(data):
        wrist_delta = data[:, [20, 21], 2] - data[:, [15], 2]
        return {'left_right_wrist_above_head_frames': (wrist_delta > 0).sum(axis=0).tolist(),
                'max_wrist_minus_head_m': wrist_delta.max(axis=0).tolist(),
                'root_xy_range_m': np.ptp(data[:, 0, :2], axis=0).tolist(),
                'root_z_range_m': float(np.ptp(data[:, 0, 2]))}
    result = {'human_sha256': hr['artifact_sha256'], 'candidate_sha256': r['output_sha256'],
              'script_sha256': sha(Path(__file__)), 'video_sha256': sha(movie),
              'fps_nominal': 20, 'frames': len(frames), 'source_frame_interval': [int(frames[0]), int(frames[-1])],
              'source_time_seconds': [float(frames[0] / 120), float(frames[-1] / 120)],
              'source_human_observations': metrics(human), 'robot_observations': metrics(robot),
              'original_caption_records': r['original_caption_records'],
              'caption_accepted': False, 'training_ready': False,
              'limitations': ['Human is direct 120-to-20 Hz sampling; robot is filtered; model and timing remain approximate.',
                              'World x/z and y/z orthographic views retain global translation.',
                              'Wrist-vs-head is a geometric proxy, not an automatic caption decision.',
                              'Motion review does not establish physical, source or training acceptance.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
