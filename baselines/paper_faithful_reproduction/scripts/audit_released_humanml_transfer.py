"""Read-only metadata audit before reusing released G1 pairs for PhyGile.

Different clip lengths do not identify FPS or prove an author dataset error.
Do not infer missing frame maps, transfer timed captions or alter active data.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import zipfile
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    paths = [a.data/'full_train.pkl', a.public/'index.csv', a.public/'HumanML3D/texts.zip', a.contract,
             *[a.data/(s+'.txt') for s in ['train', 'val', 'test']]]
    hashes = {str(p.resolve()): sha(p) for p in paths}
    if sha(a.data/'full_train.pkl') != '53a3fff8c2928d3e36b06b1b8bce1148675854ff3c5ef5bad0e56f295abe302d':
        raise ValueError('Only the already authenticated/retrieved pinned release pickle is accepted')
    # Deserializes only the previously SHA-verified official local release.
    with (a.data/'full_train.pkl').open('rb') as f:
        payload = pickle.load(f)
    data = payload['data_dict']
    if set(data) != set(payload['name_list']) or len(data) != len(payload['length_list']):
        raise ValueError('Release index inconsistency')
    split_by_name = {}
    for split in ['train', 'val', 'test']:
        for name in (a.data/(split+'.txt')).read_text().splitlines():
            if name in split_by_name:
                raise ValueError('Release split overlap')
            split_by_name[name] = split
    if set(data) != set(split_by_name):
        raise ValueError('Release split coverage incomplete')
    rows = {Path(r['new_name']).stem: r for r in csv.DictReader((a.public/'index.csv').open())}
    contract = a.contract.read_text(encoding='utf-8')
    if 'datasets use 60 Hz' not in contract:
        raise ValueError('Declared release clock must be read from the pinned primary contract')
    groups, ratio_by_subset = collections.defaultdict(set), collections.defaultdict(list)
    clips, notes = [], collections.Counter()
    with zipfile.ZipFile(a.public/'HumanML3D/texts.zip') as text_zip:
        members = {Path(n).stem: n for n in text_zip.namelist() if n.endswith('.txt')}
        for name, entry in data.items():
            motion = np.asarray(entry['motion'])
            if motion.ndim != 2 or motion.shape[1] != 36 or entry['length'] != len(motion) or not np.isfinite(motion).all():
                raise ValueError('Invalid source release tensor')
            base = name[1:] if name.startswith('M') else name
            if not re.fullmatch(r'\d{6}', base) or base not in rows:
                notes['unmapped_id'] += 1
                continue
            source = rows[base]
            source_group = source['source_path']
            subset = source_group.split('/')[2]
            groups[source_group].add(split_by_name[name])
            annotations = collections.defaultdict(set)
            if name in members:
                for line in text_zip.read(members[name]).decode().splitlines():
                    fields = line.split('#')
                    if len(fields) != 4:
                        continue
                    first, last = float(fields[2]), float(fields[3])
                    kind = 'whole' if first == 0 and last == 0 else 'timed'
                    annotations[' '.join(fields[0].split())].add(kind)
            timed = unmatched = 0
            for text in entry['text']:
                kinds = annotations.get(' '.join(text['caption'].split()), set())
                timed += 'timed' in kinds
                unmatched += not kinds
            notes['released_caption_entries'] += len(entry['text'])
            notes['captions_matching_an_original_timed_annotation'] += timed
            notes['captions_not_matched_by_literal_whitespace_normalization'] += unmatched
            notes['clips_with_timed_caption_matches'] += timed > 0
            start, end = int(source['start_frame']), int(source['end_frame'])
            original_frames = end-start if end >= 0 else None
            ratio = len(motion)/original_frames if original_frames and original_frames > 0 else None
            if ratio is not None:
                ratio_by_subset[subset].append(ratio)
            clips.append({'release_id': name, 'release_split': split_by_name[name],
                'source_group_sha256': hashlib.sha256(source_group.encode()).hexdigest(),
                'released_frames': len(motion), 'declared_release_fps': 60.,
                'index_crop_frames_nominal_20hz': original_frames,
                'frame_count_ratio_diagnostic_only': ratio,
                'timed_caption_matches': timed, 'unmatched_captions': unmatched,
                'interval_identity_verified': False, 'eligible_for_automatic_phygile_transfer': False})
    overlaps = {g: s for g, s in groups.items() if len(s) > 1}
    overlap_hashes = {hashlib.sha256(g.encode()).hexdigest() for g in overlaps}
    if any(sha(p) != h for p, h in hashes.items()):
        raise ValueError('Read-only inputs changed')
    result = {'status': 'RELEASE_TRANSFER_METADATA_AUDITED', 'clips': len(data), 'mapped_clips': len(clips),
        'declared_release_fps': 60., 'clock_inferred_from_lengths': False, 'data_modified': False,
        'unique_original_recordings': len(groups), 'cross_release_split_recordings': len(overlaps),
        'mapped_entries_in_cross_split_recordings': sum(r['source_group_sha256'] in overlap_hashes for r in clips),
        'counts': dict(notes), 'frame_ratio_by_subset': {s: {'clips': len(v),
            'min': float(min(v)), 'median': float(np.median(v)), 'max': float(max(v))}
            for s, v in ratio_by_subset.items()},
        'transfer_status': 'QUARANTINED_PENDING_PER_CLIP_CLOCK_INTERVAL_AND_SOURCE_SPLIT_MAPPING',
        'interpretation': 'The release is genuinely motion/text paired according to its source. '
            'This audit does not assert it is invalid for its own task. Its 60Hz contract, lengths '
            'and per-entry split cannot silently substitute the original HumanML20Hz/raw-source protocol. '
            'Timed-caption matches need explicit interval verification; no inferred FPS or data rewrite.',
        'inputs': hashes, 'inputs_unchanged': True, 'script_sha256': sha(__file__)}
    (a.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (a.output/'clip_metadata.json').write_text(json.dumps(clips, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['inputs', 'frame_ratio_by_subset']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'public', 'contract', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
