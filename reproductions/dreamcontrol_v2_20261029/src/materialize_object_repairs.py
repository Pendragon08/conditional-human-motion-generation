"""Write an explicit derived copy for previously audited quaternion anomalies.

Input NPZ SHA-256 must match the report. The original ZIP is never modified.
The output is not retargeted or validated as a train-ready dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np

from normalize_object_quaternion import normalize_object_quaternions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('witness', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    with args.witness.open(encoding='utf-8') as f:
        witness = json.load(f)
    if witness['archive'] != args.archive.name or witness['refused']:
        raise ValueError('witness archive does not match or includes unresolved errors')
    records = witness['repairs']
    if not records:
        raise ValueError('witness has no repairs')
    # Build everything in memory and validate all input before creating output.
    outputs, names = [], set()
    with zipfile.ZipFile(args.archive) as archive:
        for record in records:
            member = record['member']
            name = PurePosixPath(member).name
            if not name.endswith('.npz') or name in names:
                raise ValueError(f'invalid or duplicated output name: {name}')
            names.add(name)
            payload = archive.read(member)
            digest = hashlib.sha256(payload).hexdigest()
            if digest != record['source_npz_sha256']:
                raise ValueError(f'source hash mismatch: {member}')
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            fixed, meta = normalize_object_quaternions(arrays['qpos'])
            if meta['frames_outside_tolerance'] <= 0:
                raise ValueError(f'no anomaly for {member}')
            arrays['qpos'] = fixed
            encoded = io.BytesIO()
            np.savez_compressed(encoded, **arrays)
            result = encoded.getvalue()
            outputs.append((name, result, {'source_member': member, 'source_npz_sha256': digest,
                                           'output_file': name, 'output_sha256': hashlib.sha256(result).hexdigest(),
                                           **meta}))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, result, _ in outputs:
        with (args.output_dir / name).open('xb') as f:
            f.write(result)
    manifest = {'archive': args.archive.name, 'derived': [x[2] for x in outputs],
                'note': 'Object quaternion normalization only; source archives unchanged. Physical validity unverified.'}
    with (args.output_dir / 'manifest.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(f'created {len(outputs)} derived files in {args.output_dir}')


if __name__ == '__main__':
    main()
