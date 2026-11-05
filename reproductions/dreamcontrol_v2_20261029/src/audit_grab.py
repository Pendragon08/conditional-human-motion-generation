"""Audit GRAB ZIP metadata without unpickling its object-valued arrays."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def audit(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    subjects, rates, intents, objects = Counter(), Counter(), Counter(), Counter()
    frames, errors = 0, []
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and m.filename.endswith('.npz')]
        for member in members:
            try:
                # GRAB stores body/object dictionaries as pickled object arrays.
                # Inspect only primitive metadata, and never deserialize those objects.
                with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as x:
                    subject = str(x['sbj_id'].item())
                    fps = float(x['framerate'].item())
                    n = int(x['n_frames'].item())
                    intent = str(x['motion_intent'].item())
                    obj = str(x['obj_name'].item())
                    if not np.isfinite(fps) or fps <= 0 or n <= 0:
                        raise ValueError('invalid framerate or frame count')
                    if not {'body', 'object', 'contact'}.issubset(x.files):
                        raise ValueError('missing expected body/object/contact fields')
                    subjects[subject] += 1
                    rates[str(fps)] += 1
                    intents[intent] += 1
                    objects[obj] += 1
                    frames += n
            except (KeyError, ValueError, OSError, zipfile.BadZipFile) as exc:
                errors.append({'member': member.filename, 'error': str(exc)})
    return {'archive': path.name, 'sha256': digest.hexdigest(), 'sequence_count': len(members),
            'metadata_valid': len(members)-len(errors), 'errors': errors,
            'subject_counts': dict(subjects), 'fps_counts': dict(rates),
            'intent_counts': dict(intents), 'object_counts': dict(objects), 'total_frames': frames,
            'scope': 'Metadata and outer ZIP read only; object arrays, model files and motions were not validated.'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f"{result['metadata_valid']}/{result['sequence_count']} sequence metadata valid")
    return int(bool(result['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
