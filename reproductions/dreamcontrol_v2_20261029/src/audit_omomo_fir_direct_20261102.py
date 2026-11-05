"""Independently check 30-to-20 Hz preview resampling using direct convolution.

Inputs are the previously fitted 30 Hz robot states, not unretargeted ground
truth. This verifies the mathematical transformation, not motion quality.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BATCH = ROOT / 'omni_variants_gpu_20261101'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def direct_resample(q):
    q = np.asarray(q, dtype=np.float64).copy()
    if q.ndim != 2 or q.shape[1] != 41 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('invalid state')
    for start in (0, 34):
        for i in range(1, len(q)):
            if np.dot(q[i-1, start:start+4], q[i, start:start+4]) < 0:
                q[i, start:start+4] *= -1
        if not np.allclose(np.linalg.norm(q[:, start:start+4], axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError('input quaternion not unit')
    n = len(q)
    slope = (q[-1] - q[0]) / (n-1)
    residual = q - (q[0] + np.arange(n)[:, None] * slope)
    kaiser = np.kaiser(61, 5.0)
    t = np.arange(-30, 31, dtype=float)
    h = np.sinc(t / 3.) / 3. * kaiser
    h = h / h.sum() * 2.
    count = ((n-1) * 2) // 3 + 1
    result = np.empty((count, 41), dtype=float)
    # A 2x upsample and 3x decimation. The centered filter has 61 taps;
    # for time k only indexes j with |3k - 2j| <= 30 contribute.
    for k in range(count):
        indices = np.arange(int(np.ceil((3*k-30)/2)), (3*k+30)//2+1)
        samples = residual[np.clip(indices, 0, n-1)].copy()
        before, after = indices < 0, indices >= n
        line = (residual[-1] - residual[0]) / (n-1)
        samples[before] = residual[0] + indices[before, None] * line
        samples[after] = residual[-1] + (indices[after, None]-(n-1)) * line
        result[k] = h[30 + 3*k - 2*indices] @ samples + q[0] + (1.5*k) * slope
    result[0] = q[0]
    if (count-1)*3 == 2*(n-1):
        result[-1] = q[-1]
    for start in (0, 34):
        norms = np.linalg.norm(result[:, start:start+4], axis=1)
        if np.any(norms < 1e-8): raise ValueError('invalid output quaternion')
        result[:, start:start+4] /= norms[:, None]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    report_path = BATCH / 'report.json'; records_path = BATCH / 'records.jsonl'
    report_hash, records_hash = sha(report_path), sha(records_path)
    report = json.loads(report_path.read_text())
    if report['records_sha256'] != records_hash: raise ValueError('record hash mismatch')
    records = [json.loads(s) for s in records_path.read_text().splitlines()]
    if a.limit is not None: records = records[:a.limit]
    if shutil.disk_usage(ROOT).free < 8*1024**3 + 64*1024**2: raise RuntimeError('disk budget')
    a.out.mkdir(exist_ok=False)
    failures = []; max_err = 0.; checked = 0
    with (a.out / 'records.jsonl').open('x', encoding='utf-8', buffering=1) as out:
        for i, row in enumerate(records):
            sid = row['sample_id']
            try:
                if row['technical_processing'] != 'success': raise ValueError('unsuccessful source record')
                p, z = BATCH / row['state30'], BATCH / row['state20']
                if sha(p) != row['state30_sha256'] or sha(z) != row['state20_sha256']:
                    raise ValueError('state hash mismatch')
                with np.load(p, allow_pickle=False) as data:
                    q = np.concatenate((data['root_wxyz_xyz'], data['joint_pos'], data['object_wxyz_xyz']), axis=1)
                    times30 = data['sample_times'].copy()
                with np.load(z, allow_pickle=False) as data:
                    got = np.concatenate((data['root_wxyz_xyz'], data['joint_pos'], data['object_wxyz_xyz']), axis=1)
                    times20 = data['sample_times'].copy(); source_frames = data['source_frame_positions'].copy()
                expected = direct_resample(q)
                if expected.shape != got.shape: raise ValueError('output shape mismatch')
                if not np.array_equal(times30, np.arange(len(q))/30.): raise ValueError('30 Hz time mismatch')
                if not np.array_equal(times20, np.arange(len(got))/20.): raise ValueError('20 Hz time mismatch')
                if not np.array_equal(source_frames, np.arange(len(got))*1.5): raise ValueError('source positions mismatch')
                err = float(np.max(np.abs(expected-got))); max_err = max(max_err, err)
                if err > 1e-8: raise ValueError(f'direct FIR mismatch {err}')
                checked += 1
                result = {'sample_id': sid, 'passed': True, 'max_abs_error': err}
            except Exception as e:
                result = {'sample_id': sid, 'passed': False, 'error': repr(e)}; failures.append(result)
            out.write(json.dumps(result, allow_nan=False)+'\n')
            if (i+1) % 100 == 0:
                print('PROGRESS', i+1, len(records), flush=True)
                if shutil.disk_usage(ROOT).free < 8*1024**3 + 64*1024**2: raise RuntimeError('disk budget')
    if sha(report_path) != report_hash or sha(records_path) != records_hash:
        raise RuntimeError('input manifest changed')
    result = {'script_sha256': sha(Path(__file__)), 'batch_report_sha256': report_hash,
              'batch_records_sha256': records_hash, 'output_records_sha256': sha(a.out/'records.jsonl'),
              'selected': len(records), 'passed': checked, 'failures': failures, 'max_abs_error': max_err,
              'training_ready': False,
              'limitations': ['Checks 30-to-20 Hz numerical resampling only, against saved fitted robot states.',
                              'Does not verify raw source identity, semantic correspondence, physics, or licensing.',
                              'Shares recorded algorithm definition but uses independent direct convolution.']}
    with (a.out/'report.json').open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
