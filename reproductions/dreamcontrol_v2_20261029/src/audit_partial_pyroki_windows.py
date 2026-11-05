"""Audit an append-only PyRoki window prefix without modifying the live run.

This verifies only the saved prefix. Resumption requires a fresh audit after
the original process stops; this output does not authorize GPU use or training.
"""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run_dir', 'source', 'urdf', 'output'):
        p.add_argument('--' + key.replace('_', '-'), required=True, type=Path)
    a = p.parse_args()
    root = a.run_dir.resolve(strict=True)
    receipt = json.loads((root / 'prepare_manifest.json').read_text())
    if sha(root / 'run.py') != receipt['solver_sha256'] or sha(a.source) != receipt['source_sha256']:
        raise ValueError('solver or source hash mismatch')
    with np.load(a.source, allow_pickle=False) as data:
        frames, times = data['source_frames'].copy(), data['source_times'].copy()
        human = data['human']
        if human.shape != (len(frames), 22, 3) or len(frames) < 8 or not np.isfinite(human).all():
            raise ValueError('invalid human source')
    if not np.all(np.diff(frames) == 1) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError('invalid source timeline')
    starts = list(range(0, len(frames) - 7, 4))
    if starts[-1] != len(frames) - 8: starts.append(len(frames) - 8)
    if len(starts) != receipt['expected_windows']: raise ValueError('unexpected expected-window count')
    data = (root / 'windows.jsonl').read_bytes()
    if not data or not data.endswith(b'\n'): raise ValueError('empty or incomplete window log snapshot')
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) > len(starts): raise ValueError('too many window rows')
    limits = {}
    for joint in ET.parse(a.urdf).getroot().findall('joint'):
        if joint.get('type') == 'continuous': limits[joint.get('name')] = (-np.inf, np.inf)
        elif joint.get('type') == 'revolute':
            lim = joint.find('limit'); limits[joint.get('name')] = (float(lim.get('lower')), float(lim.get('upper')))
    if len(limits) != 27: raise ValueError('expected 27 URDF joints')
    covered = set(); max_violation = 0.; max_overlap_q = 0.; max_overlap_root = 0.; prev = None
    for i, row in enumerate(rows):
        expected = np.arange(starts[i] + 1, starts[i] + 8)
        if row['window'] != i or row['sample_indices'] != expected.tolist():
            raise ValueError('noncontiguous prefix or unexpected indices')
        artifact = (root / row['artifact']).resolve(strict=True)
        if not artifact.is_relative_to(root) or sha(artifact) != row['artifact_sha256']:
            raise ValueError('window path or hash mismatch')
        with np.load(artifact, allow_pickle=False) as f:
            ix = f['sample_indices'].copy(); q = f['joint_pos'].copy(); base = f['root_wxyz_xyz'].copy()
            names = [str(n) for n in f['joint_names']]
            if not np.array_equal(f['source_frames'], frames[ix]) or not np.array_equal(f['source_times'], times[ix]):
                raise ValueError('source frame mapping differs')
        if (not np.array_equal(ix, expected) or q.shape != (7, 27) or base.shape != (7, 7) or
            not np.isfinite(q).all() or not np.isfinite(base).all() or
            not np.allclose(np.linalg.norm(base[:, :4], axis=1), 1., rtol=0, atol=1e-4) or
            len(set(names)) != 27 or set(names) != set(limits)):
            raise ValueError('invalid window state')
        low = np.array([limits[n][0] for n in names]); high = np.array([limits[n][1] for n in names])
        max_violation = max(max_violation, float(np.maximum(np.maximum(low - q, q - high), 0).max()))
        covered.update(ix.tolist())
        if prev is not None:
            common = sorted(set(prev['ix']).intersection(ix.tolist()))
            if row['overlap']['sample_indices'] != common or names != prev['names']:
                raise ValueError('overlap or joint ordering mismatch')
            for index in common:
                old = prev['ix'].index(index); new = int(np.flatnonzero(ix == index)[0])
                max_overlap_q = max(max_overlap_q, float(np.max(np.abs(prev['q'][old] - q[new]))))
                max_overlap_root = max(max_overlap_root, float(np.linalg.norm(prev['base'][old, 4:] - base[new, 4:])))
        elif row['overlap'] is not None: raise ValueError('first window has an overlap')
        prev = {'ix': ix.tolist(), 'q': q, 'base': base, 'names': names}
    if covered != set(range(1, max(covered) + 1)): raise ValueError('prefix coverage has gaps')
    result = {'source_sha256': sha(a.source), 'urdf_sha256': sha(a.urdf),
              'solver_sha256': sha(root / 'run.py'), 'prefix_log_sha256': hashlib.sha256(data).hexdigest(),
              'prefix_windows': len(rows), 'expected_windows': len(starts),
              'covered_indices': [min(covered), max(covered)], 'max_joint_violation_rad': max_violation,
              'max_overlap_joint_rad': max_overlap_q, 'max_overlap_root_m': max_overlap_root,
              'training_ready': False,
              'limitations': ['Snapshot can become stale as solver runs; re-audit after process exits.',
                              'No independent optimization, GPU equivalence, source identity, contact or physical quality.']}
    with a.output.open('x', encoding='utf8') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'prefix_windows': len(rows), 'expected_windows': len(starts),
                      'max_joint_violation_rad': max_violation}), flush=True)


if __name__ == '__main__': main()
