"""Check exact motion and caption overlap in a trusted 29-DoF corpus split.

Pickle loading can execute code; use only an explicitly trusted local corpus.
Identical captions do not by themselves prove motion leakage; distinct hashes
do not prove different underlying source recordings.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def audit(corpus_path: Path, split_dir: Path) -> dict:
    with corpus_path.open('rb') as f:
        data = pickle.load(f)['data_dict']  # Explicitly trusted source only.
    splits, id_owner = {}, {}
    for name in ('train', 'val', 'test'):
        ids = [s.strip() for s in (split_dir/(name+'.txt')).read_text(encoding='utf-8').splitlines() if s.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f'duplicate sample ID in {name}')
        for key in ids:
            if key in id_owner:
                raise ValueError(f'sample ID {key} is present in multiple splits')
            if key not in data:
                raise ValueError(f'sample ID {key} is absent from corpus')
            id_owner[key] = name
        splits[name] = ids
    motions, captions = defaultdict(list), defaultdict(list)
    no_caption = []
    for key, split in id_owner.items():
        item = data[key]; motion = np.asarray(item['motion'])
        if motion.ndim != 2 or motion.shape[1] != 36 or not np.isfinite(motion).all():
            raise ValueError(f'invalid motion {key}')
        h = hashlib.sha256()
        h.update(str(motion.shape).encode()); h.update(motion.dtype.str.encode()); h.update(np.ascontiguousarray(motion).tobytes())
        motions[h.hexdigest()].append((split, key))
        normalized = {' '.join(caption['caption'].casefold().split()) for caption in item.get('text', [])
                      if isinstance(caption, dict) and isinstance(caption.get('caption'), str) and caption['caption'].strip()}
        if not normalized: no_caption.append(key)
        for text in normalized:
            captions[hashlib.sha256(text.encode('utf-8')).hexdigest()].append((split, key))
    def cross_groups(mapping):
        return [values for values in mapping.values() if len({x[0] for x in values}) > 1]
    motion_cross, caption_cross = cross_groups(motions), cross_groups(captions)
    return {'source': str(corpus_path), 'split_counts': {k:len(v) for k,v in splits.items()},
            'unassigned_corpus_samples': len(set(data)-set(id_owner)), 'no_caption_count': len(no_caption),
            'exact_motion_cross_split_groups': len(motion_cross),
            'exact_motion_cross_split_examples': [list(x) for x in motion_cross[:20]],
            'identical_caption_cross_split_groups': len(caption_cross),
            'note': 'Exact hashes only. Repeated captions alone do not prove leakage; hidden parent-source relations remain unverified.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trusted-corpus', type=Path, required=True)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args(); result=audit(args.trusted_corpus, args.split_dir)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps({k:result[k] for k in ('split_counts','exact_motion_cross_split_groups','identical_caption_cross_split_groups')}))


if __name__ == '__main__':
    main()
