"""Render continuous human/G1 comparisons for the four remaining retained clips.

The fifth retained MoSh clip already has a separately audited comparison. All
source identities, semantic reviews and physical validation remain conditional.
"""
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from g1_fk import URDFFK

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
INPUT = ROOT / 'five_conservative_preview_20261031'
ASSOC = ROOT / 'eight_caption_alignment_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
HERE = Path(__file__).resolve().parent
CHAINS = ((0, 2, 5, 8, 11), (0, 1, 4, 7, 10), (0, 3, 6, 9, 12, 15),
          (9, 14, 17, 19, 21), (9, 13, 16, 18, 20))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def interpolate_points(source_times, points, target_times):
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if (points.shape != (len(source_times), 22, 3) or not np.isfinite(points).all() or
        not np.isfinite(source_times).all() or not np.isfinite(target_times).all() or
        np.any(np.diff(source_times) <= 0) or np.any(np.diff(target_times) <= 0) or
        target_times[0] < source_times[0] - 1e-9 or target_times[-1] > source_times[-1] + 1e-9):
        raise ValueError('invalid or out-of-bounds source timeline')
    flat = points.reshape(len(source_times), -1)
    return np.column_stack([np.interp(target_times, source_times, flat[:, j]) for j in range(66)]).reshape(-1, 22, 3)


def human_data(ident, target_times):
    if ident.startswith('ACCAD_'):
        folder = ROOT / 'accad_004347_fullrate_20261031'
        report_path, file = folder / 'human_report.json', folder / 'human_120hz.npz'
        report = json.loads(report_path.read_text())
        if sha(file) != report['artifact_sha256']:
            raise ValueError('ACCAD witness changed')
        with np.load(file, allow_pickle=False) as f:
            source_times, points = f['source_times'].copy(), f['human'].copy()
        basis = 'declared 120 Hz; linear positional interpolation'
    else:
        folder = (ROOT / 'humanml_mirror_time_witness_v2_20261030' if ident == '000008'
                  else ROOT / 'ssm_three_human_candidates_20261031')
        report_path, file = folder / 'report.json', folder / (ident + '_joint_witness.npz')
        report = json.loads(report_path.read_text())
        witness = next(x for x in report['witnesses'] if x['id'] == ident)
        if sha(file) != witness['artifact_sha256']:
            raise ValueError('SSM witness changed')
        with np.load(file, allow_pickle=False) as f:
            source_times, hm = f['source_times'].copy(), f['original_joints_humanml'].copy()
        points = hm[..., [0, 2, 1]].copy()
        points[..., 0] *= -1
        basis = 'reference-derived times; HumanML inverse axis mapping; linear positional interpolation'
    return interpolate_points(source_times, points, target_times), {'path': str(file), 'sha256': sha(file),
            'report_sha256': sha(report_path), 'time_basis': basis}


def main():
    parent = json.loads((INPUT / 'report.json').read_text())
    if (sha(INPUT / 'manifest.json') != parent['manifest_sha256'] or
        sha(INPUT / 'state_manifest.json') != parent['state_manifest_sha256']):
        raise ValueError('preview manifests changed')
    records = json.loads((INPUT / 'manifest.json').read_text())['records']
    states = {x['sample_id']: x for x in json.loads((INPUT / 'state_manifest.json').read_text())['records']}
    association = json.loads((ASSOC / 'caption_association.json').read_text())
    audit = json.loads((ASSOC / 'independent_audit.json').read_text())
    if sha(ASSOC / 'caption_association.json') != audit['association_sha256']:
        raise ValueError('caption association changed')
    captions = {x['sample_id']: x for x in association['cases']}
    fk = URDFFK(URDF); outputs = []
    for rec in records:
        ident = rec['sample_id']
        if ident == 'MOSH_000798_crop':
            continue
        state = states[ident]; state_path = INPUT / state['state_path']
        if sha(state_path) != state['state_sha256'] or sha(INPUT / rec['feature_path']) != rec['feature_sha256']:
            raise ValueError('state or feature changed')
        with np.load(state_path, allow_pickle=False) as f:
            roots, q, times, names = (f[x].copy() for x in ('root_wxyz_xyz', 'joint_pos', 'sample_times', 'joint_names'))
        if roots.shape != (len(times), 7) or q.shape != (len(times), 27) or not np.allclose(np.diff(times), .05, atol=1e-8, rtol=0):
            raise ValueError('invalid robot timeline')
        robot = fk.forward(roots[:, 4:], roots[:, [1, 2, 3, 0]], q, list(names))
        human, provenance = human_data(ident, times)
        if human.shape != robot.shape or human.shape != (len(times), 22, 3):
            raise ValueError('invalid pose arrays')
        desc = captions[ident]
        if desc['feature_sha256'] != rec['feature_sha256'] or rec['caption'] not in [x['caption'] for x in desc['captions']]:
            raise ValueError('caption correspondence changed')
        positions = np.concatenate([human.reshape(-1, 3), robot.reshape(-1, 3)])
        mn, mx = positions.min(0), positions.max(0); center = (mn + mx) / 2
        radius = float(max((mx - mn).max() * .58, .4))
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
        views = ((0, 2, 'x/z'), (1, 2, 'y/z')); lines = []
        for row, label in enumerate(('Source human fit', 'Retargeted G1')):
            pair = []
            for col, (x, y, view) in enumerate(views):
                ax = axes[row, col]
                ax.set(xlim=(center[x]-radius, center[x]+radius), ylim=(center[y]-radius, center[y]+radius), aspect='equal')
                ax.set_title(label + ' ' + view, fontsize=8); ax.grid(alpha=.2)
                ax.set_xlabel('world ' + 'xyz'[x] + ' m', fontsize=7); ax.set_ylabel('world ' + 'xyz'[y] + ' m', fontsize=7)
                pair.append([ax.plot([], [], '-o', markersize=2, linewidth=1.5)[0] for _ in CHAINS])
            lines.append(pair)
        fig.suptitle(ident + ' | ' + rec['split'] + '\n' + '\n'.join(textwrap.wrap(rec['caption'], 100)), fontsize=8)
        stamp = fig.text(.5, .01, '', ha='center', fontsize=8)
        target = HERE / (ident + '.mp4')
        if target.exists(): raise FileExistsError(target)
        writer = FFMpegWriter(fps=20, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-threads', '1'])
        with writer.saving(fig, str(target), dpi=100):
            for i in range(len(times)):
                for row, values in enumerate((human, robot)):
                    for col, (x, y, _) in enumerate(views):
                        for line, chain in zip(lines[row][col], CHAINS):
                            line.set_data(values[i, list(chain), x], values[i, list(chain), y])
                stamp.set_text(f'Nominal source time {times[i]:.2f} s | sample {i+1}/{len(times)}')
                writer.grab_frame()
        plt.close(fig)
        outputs.append({'sample_id': ident, 'split': rec['split'], 'feature_sha256': rec['feature_sha256'],
                        'state_sha256': state['state_sha256'], 'source_evidence': provenance,
                        'caption_member_sha256': desc['caption_member_sha256'],
                        'caption_records': [{'caption': c['caption'], 'raw_tags': c['raw_tags']} for c in desc['captions']],
                        'chosen_caption': rec['caption'], 'video_sha256': sha(target), 'frames': len(times),
                        'source_sample_time_first_last': [float(times[0]), float(times[-1])],
                        'human_aligned_points_sha256': hashlib.sha256(np.ascontiguousarray(human, dtype='<f8').tobytes()).hexdigest(),
                        'robot_points_sha256': hashlib.sha256(np.ascontiguousarray(robot, dtype='<f8').tobytes()).hexdigest(),
                        'caption_accepted': False, 'training_ready': False})
        print(json.dumps({'finished': ident, 'frames': len(times)}), flush=True)
    if len(outputs) != 4:
        raise ValueError('unexpected retained subset')
    report = {'manifest_sha256': sha(INPUT / 'manifest.json'), 'state_manifest_sha256': sha(INPUT / 'state_manifest.json'),
              'caption_association_sha256': sha(ASSOC / 'caption_association.json'), 'urdf_sha256': sha(URDF),
              'script_sha256': sha(Path(__file__)), 'cases': outputs, 'training_ready': False,
              'limitations': ['Human fit and robot FK are approximate; positional interpolation is not pose resampling.',
                              'Fixed world axes and translation preserved; matching views do not establish physical contact.',
                              'No independent full video review, actual FPS, caption semantics or training acceptance.']}
    with (HERE / 'report.json').open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    print(json.dumps({'finished_all': len(outputs), 'training_ready': False}), flush=True)


if __name__ == '__main__':
    main()
