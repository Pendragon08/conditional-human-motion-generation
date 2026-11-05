"""Generate full-frame, nominal-time human/G1 contact sheets for manual review.

This is an approximate visual aid, not semantic, clock, or physical acceptance.
"""
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE = ROOT / 'kit_two_selected_20261102'
OUT = BASE / 'pair_review_contact_sheets_20261102'
CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15),
          (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if shutil.disk_usage(BASE).free < 8 * 1024**3 + 64 * 1024**2:
        raise RuntimeError('insufficient disk budget')
    OUT.mkdir(exist_ok=False)
    result = []
    for sid in ('004271', '007555'):
        human_path = BASE / ('human_' + sid) / 'human_20hz_declared.npz'
        robot_path = BASE / 'robot_263_preview' / (sid + '.npz')
        hr = json.loads((human_path.parent / 'report.json').read_text())
        rr = json.loads((robot_path.parent / (sid + '_report.json')).read_text())
        hh, rh = sha(human_path), sha(robot_path)
        if hh != hr['artifact_sha256'] or rh != rr['output_sha256']:
            raise ValueError('input artifact hash mismatch')
        with np.load(human_path, allow_pickle=False) as f:
            human = f['human'].copy(); hf = f['source_frames'].copy()
        with np.load(robot_path, allow_pickle=False) as f:
            robot = f['points_z_up_world'].copy(); rf = f['source_frames'].copy()
        indices = np.searchsorted(hf, rf)
        if (len(rf) != 95 or not np.array_equal(hf[indices], rf)
                or not np.isfinite(human).all() or not np.isfinite(robot).all()):
            raise ValueError('invalid frame alignment')
        human = human[indices]
        if human.shape != robot.shape or robot.shape != (95, 22, 3):
            raise ValueError('unexpected joints')
        points = np.concatenate((human.reshape(-1, 3), robot.reshape(-1, 3)))
        low, high = points.min(0), points.max(0)
        center = (low + high) / 2
        radius = max(float((high - low).max()) * .56, .5)
        sheets = []
        for start in range(0, len(rf), 8):
            if shutil.disk_usage(BASE).free < 8 * 1024**3 + 64 * 1024**2:
                raise RuntimeError('insufficient disk budget')
            fig, axes = plt.subplots(4, 4, figsize=(12, 10), constrained_layout=True)
            fig.suptitle(sid + '   blue: human SMPL-X  /  orange: G1   |   declared clock only')
            for j in range(8):
                i = start + j
                for v, (x, label) in enumerate(((0, 'x/z'), (1, 'y/z'))):
                    ax = axes[j // 2, (j % 2) * 2 + v]
                    ax.set(xlim=(center[x]-radius, center[x]+radius),
                           ylim=(center[2]-radius, center[2]+radius), aspect='equal')
                    ax.set_xticks([]); ax.set_yticks([])
                    if i >= len(rf):
                        ax.axis('off'); continue
                    for data, color in ((human, 'tab:blue'), (robot, 'tab:orange')):
                        for chain in CHAINS:
                            ax.plot(data[i, list(chain), x], data[i, list(chain), 2],
                                    '-', color=color, linewidth=1.1, alpha=.8)
                    ax.set_title(f'{label} source frame {rf[i]}', fontsize=8)
            target = OUT / f'{sid}_{start:03d}.png'
            fig.savefig(target, dpi=100); plt.close(fig)
            sheets.append({'file': target.name, 'sha256': sha(target),
                           'first_index': start, 'last_index': min(start+7, 94)})
        if sha(human_path) != hh or sha(robot_path) != rh:
            raise RuntimeError('input changed during review rendering')
        result.append({'id': sid, 'human_sha256': hh, 'robot_sha256': rh,
                       'aligned_frames': len(rf), 'source_frames_first_last': [int(rf[0]), int(rf[-1])],
                       'sheets': sheets, 'caption_accepted': False})
    report = {'script_sha256': sha(Path(__file__)), 'cases': result,
              'training_ready': False,
              'limitations': ['SMPL-X human and G1 approximations in world coordinates.',
                              'Declared 120 Hz only; actual source clock and identity unresolved.',
                              'Visual alignment cannot establish physical feasibility or caption acceptance.']}
    with (OUT / 'report.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'output': str(OUT), 'cases': len(result), 'frames_per_case': 95}), flush=True)


if __name__ == '__main__':
    main()
