"""Deterministic group-preserving split of explicit motion provenance records.

Callers must establish each group from actual provenance, including related
segments and augmentations. Hashing groups cannot discover hidden relatives.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def assign_splits(records: list[dict], *, seed: str, train_fraction: float = .8,
                  val_fraction: float = .1) -> list[dict]:
    if not isinstance(seed, str) or not seed:
        raise ValueError('seed must be a nonempty string')
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and
            train_fraction + val_fraction < 1):
        raise ValueError('train and validation fractions must leave a test fraction')
    result, seen = [], set()
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError('every record must be an object')
        sample_id, group = rec.get('sample_id'), rec.get('group_id')
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError('sample_id must be a nonempty string')
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f'missing provenance group for {sample_id}')
        if sample_id in seen:
            raise ValueError(f'duplicate sample_id: {sample_id}')
        if 'split' in rec:
            raise ValueError(f'input record already has a split: {sample_id}')
        seen.add(sample_id)
        raw = json.dumps([seed, group], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        value = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64
        split = 'train' if value < train_fraction else ('val' if value < train_fraction + val_fraction else 'test')
        result.append({**rec, 'split': split})
    if not result:
        raise ValueError('no records provided')
    return result


def audit_assignments(records: list[dict]) -> None:
    """Reject repeated samples or a declared group crossing split boundaries."""
    seen, group_splits = set(), {}
    for rec in records:
        sample, group, split = rec.get('sample_id'), rec.get('group_id'), rec.get('split')
        if not all(isinstance(x, str) and x.strip() for x in (sample, group, split)):
            raise ValueError('each assignment needs nonempty sample_id, group_id, split')
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'invalid split {split!r}')
        if sample in seen:
            raise ValueError(f'duplicate sample_id: {sample}')
        seen.add(sample)
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError(f'provenance group {group!r} crosses splits')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='JSON object containing explicit records')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--train-fraction', type=float, default=.8)
    parser.add_argument('--val-fraction', type=float, default=.1)
    args = parser.parse_args()
    with args.input.open(encoding='utf-8') as f:
        records = json.load(f)['records']
    assigned = assign_splits(records, seed=args.seed, train_fraction=args.train_fraction,
                             val_fraction=args.val_fraction)
    audit_assignments(assigned)
    result = {'seed': args.seed, 'train_fraction': args.train_fraction, 'val_fraction': args.val_fraction,
              'counts': dict(Counter(x['split'] for x in assigned)), 'records': assigned,
              'limitation': 'Splits are valid only if group_id captures all related source motions.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
