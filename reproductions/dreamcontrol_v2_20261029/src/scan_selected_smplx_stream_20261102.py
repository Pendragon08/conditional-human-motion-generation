"""Read-only selected SMPL-X tar inventory against nominal HumanML indices.

Never extracts archive files. Matching is by names, not confirmed identity. Results
do not grant data, text, time, physical or training acceptance.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tarfile
import time
import zipfile

import numpy as np


def normalized(name):
    return re.sub(r'[^a-z0-9]', '', name.casefold())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', required=True)
    p.add_argument('--dataset', required=True, choices=['BMLmovi', 'Eyes_Japan_Dataset'])
    p.add_argument('--mapping', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    archive, mapping, out = Path(args.archive), Path(args.mapping), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    stat_before = archive.stat()
    expected = {}
    mapping_bytes = mapping.read_bytes()
    for line in mapping_bytes.splitlines():
        row = json.loads(line)
        source = row['source_path'].replace('\\', '/')
        parts = source.split('/')
        if len(parts) < 4 or parts[-3] != args.dataset or not parts[-1].endswith('_poses.npy'):
            continue
        key = (parts[-2].casefold(), normalized(parts[-1][:-len('_poses.npy')]))
        expected.setdefault(key, []).append({'motion_id': row['motion_id'], 'path': source,
                                            'start': row['index_start_20'], 'end': row['index_end_20']})
    counts = {'tar_members': 0, 'stageii_members': 0, 'matched_members': 0,
              'matched_id_rows': 0, 'covered_id_rows': 0, 'short_id_rows': 0,
              'unresolved_id_rows': 0, 'member_errors': 0}
    started = time.time()
    seen = set()
    with (out / 'records.jsonl').open('x', encoding='utf-8', buffering=1) as records, \
         (out / 'progress.jsonl').open('x', encoding='utf-8', buffering=1) as progress, \
         archive.open('rb') as raw:
        progress.write(json.dumps({'event': 'start', 'dataset': args.dataset, 'archive': str(archive),
            'archive_size': stat_before.st_size, 'archive_mtime_ns': stat_before.st_mtime_ns,
            'mapping_sha256': hashlib.sha256(mapping_bytes).hexdigest(),
            'expected_nominal_keys': len(expected), 'started_unix': started})+'\n')
        with tarfile.open(fileobj=raw, mode='r|bz2') as tar:
            for member in tar:
                counts['tar_members'] += 1
                if counts['tar_members'] % 250 == 0:
                    if shutil.disk_usage(out).free < 512*1024*1024:
                        raise RuntimeError('output filesystem free space under 512 MiB')
                    progress.write(json.dumps({'event':'progress', 'counts':counts,
                        'compressed_bytes_read_approx':raw.tell(), 'elapsed_s':round(time.time()-started, 1)})+'\n')
                name = member.name.replace('\\', '/')
                parts = name.split('/')
                if not member.isfile() or len(parts) != 3 or parts[0] != args.dataset or not parts[-1].endswith('_stageii.npz'):
                    continue
                counts['stageii_members'] += 1
                key = (parts[1].casefold(), normalized(parts[2][:-len('_stageii.npz')]))
                if key not in expected:
                    continue
                matches = expected[key]
                record = {'member': name, 'member_size': member.size, 'name_key': list(key),
                          'nominal_rows': matches, 'duplicate_key': key in seen,
                          'training_accepted': False}
                seen.add(key)
                counts['matched_members'] += 1
                counts['matched_id_rows'] += len(matches)
                try:
                    if member.size > 64*1024*1024:
                        raise ValueError('member exceeds 64 MiB per-record memory limit')
                    blob = tar.extractfile(member).read()
                    if len(blob) != member.size:
                        raise ValueError('truncated member')
                    record['member_sha256'] = hashlib.sha256(blob).hexdigest()
                    with zipfile.ZipFile(io.BytesIO(blob)) as z:
                        with z.open('poses.npy') as pose:
                            version = np.lib.format.read_magic(pose)
                            if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(pose)
                            elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(pose)
                            else: raise ValueError('unsupported numpy header version')
                        if len(shape) != 2 or dtype.hasobject:
                            raise ValueError('unsupported poses array')
                        with z.open('mocap_frame_rate.npy') as f:
                            fps = float(np.lib.format.read_array(f, allow_pickle=False))
                    if not math.isfinite(fps) or fps < 20:
                        raise ValueError('invalid declared fps')
                    stride = int(fps / 20)
                    count = len(range(0, shape[0], stride))
                    record.update({'poses_shape': list(shape), 'declared_fps': fps,
                                   'integer_stride': stride, 'nominal_20hz_count': count})
                    for row in matches:
                        try:
                            start, end = int(row['start']), int(row['end'])
                            if start < 0 or end <= start: raise ValueError('invalid interval')
                            row['interval_within_nominal_count'] = end <= count
                            counts['covered_id_rows' if end <= count else 'short_id_rows'] += 1
                        except ValueError:
                            row['interval_within_nominal_count'] = None
                            counts['unresolved_id_rows'] += 1
                except Exception as e:
                    counts['member_errors'] += 1
                    counts['unresolved_id_rows'] += len(matches)
                    record['error'] = str(e)
                records.write(json.dumps(record, ensure_ascii=False)+'\n')
        stat_after = archive.stat()
        summary = {'event': 'complete', 'dataset': args.dataset, 'counts': counts, 'expected_nominal_keys': len(expected),
                   'matched_unique_keys': len(seen), 'elapsed_s':round(time.time()-started, 1),
                   'archive_size_unchanged':stat_after.st_size == stat_before.st_size,
                   'archive_mtime_unchanged':stat_after.st_mtime_ns == stat_before.st_mtime_ns,
                   'limitations':['Name matching is not identity proof; normalized collisions require review.',
                    'Header FPS is declared; interval counts do not prove true timing or text semantics.',
                    'SMPL-X and SMPL+H are different fitted models; no reference equivalence or physical acceptance.',
                    'Archive SHA-256 not computed and no training acceptance granted.']}
        progress.write(json.dumps(summary, ensure_ascii=False)+'\n')
        with (out/'summary.json').open('x',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)


if __name__ == '__main__':
    main()
