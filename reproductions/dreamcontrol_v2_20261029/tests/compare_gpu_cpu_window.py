"""Compare one actual GPU/CPU PyRoki window; not a physical-quality gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu-run', required=True, type=Path)
    ap.add_argument('--gpu-run', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    a = ap.parse_args()
    prepared = json.loads((a.gpu_run / 'prepare_manifest.json').read_text())
    report = json.loads((a.gpu_run / 'report.json').read_text())
    cpu = a.cpu_run / 'window_00.npz'; gpu = a.gpu_run / 'window_00.npz'
    assert sha(cpu) == prepared['cpu_window_sha256']
    assert sha(a.gpu_run / 'run.py') == prepared['gpu_solver_sha256'] == report['script_sha256']
    rows = [json.loads(x) for x in (a.gpu_run / 'windows.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and sha(gpu) == rows[0]['artifact_sha256']
    with np.load(cpu, allow_pickle=False) as c, np.load(gpu, allow_pickle=False) as g:
        for key in ('sample_indices', 'source_frames', 'source_times', 'joint_names'):
            assert np.array_equal(c[key], g[key])
        ca, ga = c['root_wxyz_xyz'], g['root_wxyz_xyz']; cq, gq = c['joint_pos'], g['joint_pos']
        assert np.isfinite(ga).all() and np.isfinite(gq).all()
        pos = float(np.linalg.norm(ca[:, 4:] - ga[:, 4:], axis=-1).max())
        angle = float((Rotation.from_quat(ca[:, [1, 2, 3, 0]]).inv() *
                       Rotation.from_quat(ga[:, [1, 2, 3, 0]])).magnitude().max())
        joint = float(np.max(np.abs(cq - gq)))
    result = {'cpu_window_sha256': sha(cpu), 'gpu_window_sha256': sha(gpu),
              'root_position_max_m': pos, 'root_rotation_max_rad': angle, 'joint_max_rad': joint,
              'comparison_policy': {'root_position_max_m': .02, 'root_rotation_max_rad': .05, 'joint_max_rad': .1},
              'within_comparison_policy': pos <= .02 and angle <= .05 and joint <= .1,
              'training_ready': False,
              'limitations': ['One window only; thresholds are engineering checks, not paper or physical criteria.',
                              'GPU/CPU optimizers may diverge on other motions.']}
    with a.output.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
