"""Audit motion headers and proposed slice ranges; no inferred timeline identity."""
import collections
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

BASE = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MAPPING = BASE / 'humanml_raw_mapping_v2_20261030/mapping.jsonl'
OUT = BASE / 'humanml_timing_header_audit_20261030'
TRIM = {'Eyes_Japan_Dataset': 60, 'MPI_HDM05': 60, 'TotalCapture': 20,
        'MPI_Limits': 20, 'Transitions_mocap': 10}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def npy_header(z, key):
    with z.open(key + '.npy') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif version in ((2, 0), (3, 0)):
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('unsupported NPY version')
        return list(shape), str(dtype)


def inspect(path):
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        if 'trans.npy' not in names or 'poses.npy' not in names:
            raise ValueError('missing trans or poses')
        trans_shape, trans_dtype = npy_header(z, 'trans')
        pose_shape, pose_dtype = npy_header(z, 'poses')
    if len(trans_shape) != 2 or trans_shape[1] != 3 or not pose_shape or pose_shape[0] != trans_shape[0]:
        raise ValueError('motion shape mismatch')
    with np.load(path, allow_pickle=False) as data:
        keys = [k for k in ('mocap_frame_rate', 'mocap_framerate') if k in data]
        if len(keys) != 1:
            raise ValueError('missing or ambiguous fps header')
        fps = float(data[keys[0]])
    if not math.isfinite(fps) or fps < 20:
        raise ValueError('invalid fps')
    stride = int(fps / 20)
    return {'fps_header': fps, 'frames': trans_shape[0], 'trans_shape':trans_shape,
            'poses_shape':pose_shape, 'trans_dtype':trans_dtype, 'poses_dtype':pose_dtype,
            'floor_stride':stride, 'floor_stride_effective_fps':fps/stride,
            'near_20hz':abs(fps/stride-20) <= 0.01}


def main():
    entries = [json.loads(line) for line in MAPPING.open(encoding='utf8')]
    candidates = [row for row in entries if row['candidate_local_path']]
    cache, results, totals, per_dataset = {}, [], collections.Counter(), collections.defaultdict(collections.Counter)
    for row in candidates:
        path = row['candidate_local_path']
        dataset = PurePosixPath(row['source_path']).parts[1]
        if path not in cache:
            try:
                cache[path] = {'status':'header_ok', **inspect(path)}
            except Exception as exc:
                cache[path] = {'status':'header_error', 'error_type':type(exc).__name__, 'error':str(exc)}
        meta = cache[path]
        result = {'motion_id':row['motion_id'], 'dataset':dataset, 'source':row['source_path'],
                  'candidate_local_path':path, 'header':meta, 'slice_status':'not_checked'}
        if meta['status'] == 'header_ok':
            length = len(range(0, meta['frames'], meta['floor_stride'])) - TRIM.get(dataset,0)
            first, stop = int(row['index_start_20']), int(row['index_end_20'])
            effective_stop = length - 1 if stop == -1 else stop
            result['available_posttrim_samples'] = length
            result['slice_status'] = 'nominal_in_bounds' if first >= 0 and first < effective_stop <= length else 'out_of_bounds'
            result['nominal_clip_samples'] = max(0, effective_stop - first)
        totals[meta['status']] += 1
        totals[result['slice_status']] += 1
        if meta['status'] == 'header_ok':
            totals['nominal_near_20hz' if meta['near_20hz'] else 'nominal_not_20hz'] += 1
        per_dataset[dataset][result['slice_status']] += 1
        results.append(result)
    OUT.mkdir(mode=0o700)
    with (OUT/'results.jsonl').open('x',encoding='utf8') as f:
        for result in results: f.write(json.dumps(result,allow_nan=False)+'\n')
    fps_by_file = collections.Counter(str(v['fps_header']) for v in cache.values() if v['status']=='header_ok')
    summary = {'mapping_sha256':digest(MAPPING), 'candidate_ids':len(candidates), 'distinct_files':len(cache),
        'fps_header_counts_distinct_files':dict(fps_by_file), 'totals_by_id':dict(totals),
        'slice_counts_by_dataset':dict(per_dataset), 'results_sha256':digest(OUT/'results.jsonl'),
        'limitations':['These are metadata and nominal slice checks on target exports, not source-reference time alignment.',
                       'Reference HumanML frames may differ from the target export even when headers and ranges look plausible.',
                       'No body model, mirror, annotation segment, data license, dynamics or training readiness validated.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__ == '__main__': main()
