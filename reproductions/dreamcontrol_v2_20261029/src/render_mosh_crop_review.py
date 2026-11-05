"""Render a hash-bound temporal contact sheet for the isolated MoSh crop."""
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15),
          (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path = HERE / 'report.json'
    candidate_path = HERE / 'candidate.npz'
    report = json.loads(report_path.read_text())
    if sha(candidate_path) != report['output_sha256']:
        raise ValueError('candidate changed')
    with np.load(candidate_path, allow_pickle=False) as f:
        points, frames = f['points_z_up_world'].copy(), f['frames_20hz'].copy()
    if points.shape != (149, 22, 3) or not np.array_equal(frames, np.arange(126, 1015, 6)):
        raise ValueError('unexpected candidate')
    ids = np.rint(np.linspace(0, len(frames) - 1, 15)).astype(int)
    centered = points - points[:, :1, :]
    radius = max(float(np.abs(centered).max()) * 1.08, .2)
    fig, axes = plt.subplots(2, len(ids), figsize=(26, 5.2), constrained_layout=True)
    for col, i in enumerate(ids):
        for row, horizontal in enumerate((0, 1)):
            ax = axes[row, col]
            for chain in CHAINS:
                ax.plot(centered[i, list(chain), horizontal], centered[i, list(chain), 2], '-o', markersize=2, linewidth=1.2)
            ax.set(xlim=(-radius, radius), ylim=(-radius, radius), aspect='equal')
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(('x' if row == 0 else 'y') + ' / z')
            if row == 0:
                ax.set_title(f'{frames[i] / 120:.2f}s', fontsize=8)
    captions = ' | '.join(x['caption'] for x in report['original_caption_records'])
    fig.suptitle('000798 nominal source time; root-centered views\n' + '\n'.join(textwrap.wrap(captions, 180)), fontsize=10)
    output = HERE / 'caption_review.png'
    if output.exists():
        raise FileExistsError(output)
    fig.savefig(output, dpi=120)
    plt.close(fig)
    index = {'candidate_sha256': sha(candidate_path), 'report_sha256': sha(report_path),
             'script_sha256': sha(Path(__file__)), 'image_sha256': sha(output),
             'indices': ids.tolist(), 'source_frames': frames[ids].tolist(),
             'limitations': '15 discrete root-centered samples; not continuous-motion, source truth or physical acceptance.',
             'caption_accepted': False, 'training_ready': False}
    with (HERE / 'caption_review.json').open('x', encoding='utf8') as f:
        json.dump(index, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(index))


if __name__ == '__main__':
    main()
