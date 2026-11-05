"""Prepare two bounded CPU retargeting candidates from existing BML witnesses.

Candidate excerpts are not accepted data, and captions for these excerpts are
not asserted. The original AMASS data, published splits and witnesses are read
only. See each generated report for approximation and provenance limits.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
WITNESS = ROOT / 'bml_overlap_encoded_20261031'
PRIOR = ROOT / 'pyroki_ssm_000008_global_20261031'
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError('unexpected solver template occurrence: ' + old)
    return text.replace(old, new)


def main():
    report = json.loads((WITNESS / 'report.json').read_text())
    audit = json.loads((WITNESS / 'audit.json').read_text())
    if (audit['status'] != 'passed' or audit['witness_sha256'] != sha(WITNESS / 'report.json')
            or audit['artifact_sha256'] != report['artifact_sha256']
            or sha(WITNESS / 'overlap.npz') != report['artifact_sha256']):
        raise ValueError('witness artifact changed')
    prior = json.loads((PRIOR / 'report.json').read_text())
    if sha(PRIOR / 'run.py') != prior['script_sha256']:
        raise ValueError('solver changed')
    template = (PRIOR / 'run.py').read_text()
    launcher = (PRIOR / 'run.sh').read_text()
    rows = []
    with np.load(WITNESS / 'overlap.npz', allow_pickle=False) as data:
        for n, label in enumerate(('S04_Expert', 'S08_Novice')):
            case = report['cases'][n]
            frames = data[f'case_{n}_source_frames'][:61].astype(np.int64)
            joints = data[f'case_{n}_joints_a'][:61].astype(np.float32)
            if joints.shape != (61, 22, 3) or not np.isfinite(joints).all() or not np.all(np.diff(frames) == 6):
                raise ValueError('unexpected BML source')
            folder = HERE / label
            folder.mkdir()
            src = folder / 'human.npz'
            with src.open('xb') as f:
                np.savez_compressed(f, human=joints, source_frames=frames, source_times=frames.astype(np.float64)/120.)
            text = template
            text = change(text, "ROOT / 'humanml_mirror_time_witness_v2_20261030/000008_joint_witness.npz'", "Path(__file__).with_name('human.npz')")
            text = change(text, '8354f592da84aa9e62b87f4ab467465cd8a43120e39d5c8ae346966a8883a0b7', sha(src))
            text = change(text, "d['original_joints_humanml']", "d['human']")
            text = change(text, 'p.shape != (43, 22, 3)', 'p.shape != (61, 22, 3)')
            text = change(text, 'human = np.stack((-p[..., 0], p[..., 2], p[..., 1]), axis=-1)', 'human = p')
            text = change(text, 'abs(fps - 20.00000762939453) > 1e-6', 'abs(fps - 20.0) > 1e-6')
            text = change(text, 'q.shape != (42, 27) or rq.shape != (42, 7)', 'q.shape != (60, 27) or rq.shape != (60, 7)')
            text = change(text, 'sample_indices=np.arange(1, 43)', 'sample_indices=np.arange(1, 61)')
            text = change(text, "'fps_reference_derived': fps", "'fps_assuming_declared_source_rate': fps")
            text = change(text, "'one candidate, marker evidence inherited'", "'BML candidate identity and declared FPS not independently verified'")
            text = change(text, "'no verified caption, split, license or physical acceptance'", "'excerpt caption and time semantics, BML-specific license, split and physical quality unaccepted'")
            with (folder / 'run.py').open('x', encoding='utf8', newline='\n') as f:
                f.write(text)
            shell = change(launcher, 'pyroki_ssm_000008_global_20261031', 'bml_global_candidates_20261031/' + label)
            with (folder / 'run.sh').open('x', encoding='utf8', newline='\n') as f:
                f.write(shell)
            rows.append({'label': label, 'associated_full_clip_ids': case['ids'],
                         'candidate_source_sha256': case['candidate_sha256'], 'model_sha256': case['model_sha256'],
                         'source_frames_first_last': [int(frames[0]), int(frames[-1])],
                         'human_sha256': sha(src), 'solver_sha256': sha(folder / 'run.py'),
                         'launcher_sha256': sha(folder / 'run.sh')})
    output = {'witness_report_sha256': sha(WITNESS/'report.json'),
              'witness_audit_sha256': sha(WITNESS/'audit.json'), 'cases': rows,
              'training_ready': False,
              'limitations': ['diagnostic excerpts, captions not yet paired or evaluated',
                              'BML declared FPS and official intermediate identity not verified',
                              'SMPL-X approximation, v1 solver, assumed floor, no physical acceptance']}
    with (HERE / 'manifest.json').open('x', encoding='utf8') as f:
        json.dump(output, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
