"""Conservative path-based HumanML-to-local-motion inventory; not temporal validation."""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

BASE = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass')
PUBLIC = BASE / 'humanml_pairing_20261021/code/public'
OUTPUT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/humanml_raw_mapping_v2_20261030')
SSM_REFERENCE = BASE / 'humanml_reference_20261021/SSM_reference_smplh/extracted/SSM_synced'
ALIASES = {'SSM_synced': ('ssm_20261021/extracted/SSM', 'SSM'),
           'ACCAD': ('recovery_20261021_v1/ACCAD/extracted/ACCAD', 'ACCAD'),
           'DFaust_67': ('recovery_20261021_v1/DFaust/extracted/DFaust', 'DFaust'),
           'HumanEva': ('recovery_20261021_v1/HumanEva/extracted/HumanEva', 'HumanEva'),
           'SFU': ('remaining_20261021/SFU/extracted/SFU', 'SFU'),
           'Transitions_mocap': ('remaining_20261021/Transitions/extracted/Transitions', 'Transitions'),
           'BMLhandball': ('remaining_20261021/BMLhandball/extracted/BMLhandball', 'BMLhandball'),
           'MPI_Limits': ('remaining_20261021/PosePrior/extracted/PosePrior', 'PosePrior'),
           'EKUT': ('remaining_20261021_v4/EKUT/extracted/EKUT', 'EKUT'),
           'MPI_mosh': ('remaining_20261021_v4/MoSh/extracted/MoSh', 'MoSh'),
           'MPI_HDM05': ('remaining_20261021_v5/HDM05/extracted/HDM05', 'HDM05')}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    with (PUBLIC / 'index.csv').open(newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    index = {Path(r['new_name']).stem: r for r in rows}
    if len(index) != len(rows):
        raise ValueError('duplicate HumanML original IDs')
    split_ids = {}
    for split in ('train', 'val', 'test'):
        for mid in (PUBLIC / 'HumanML3D' / (split + '.txt')).read_text().splitlines():
            if mid in split_ids:
                raise ValueError('duplicate split ID')
            split_ids[mid] = split
    if set(split_ids) != set((PUBLIC / 'HumanML3D/all.txt').read_text().splitlines()):
        raise ValueError('split union differs from all')
    counters = collections.defaultdict(collections.Counter)
    recordings = collections.defaultdict(set)
    records = []
    dir_cache = {}
    for mid, split in sorted(split_ids.items()):
        canonical = mid[1:] if mid.startswith('M') else mid
        row = index[canonical]
        path = PurePosixPath(row['source_path'])
        if len(path.parts) < 4 or path.parts[0] != 'pose_data' or '..' in path.parts:
            raise ValueError('unsafe source path')
        dataset = path.parts[1]
        candidate = None
        if dataset == 'humanact12':
            status = 'separate_humanact12_not_searched'
        elif dataset not in ALIASES:
            status = 'source_root_not_in_current_scan'
        elif not path.name.endswith('_poses.npy'):
            status = 'unsupported_source_suffix'
        else:
            root = BASE / ALIASES[dataset][0]
            rel = PurePosixPath(*path.parts[2:])
            filename = rel.name[:-len('_poses.npy')]
            variants = [root / rel.parent / (filename + '_stageii.npz'),
                        root / rel.parent / (filename + '_poses.npz')]
            matches = [p for p in variants if p.is_file()]
            status = 'candidate_path_match' if len(matches) == 1 else ('ambiguous' if matches else 'missing_in_known_root')
            candidate = str(matches[0]) if len(matches) == 1 else None
            if not matches and (root / rel.parent).is_dir():
                parent = root / rel.parent
                if parent not in dir_cache:
                    dir_cache[parent] = list(parent.glob('*.npz'))
                def normalized(s):
                    return re.sub(r'\s+', ' ', s.replace('_', ' ')).strip().casefold()
                equivalent = []
                for p in dir_cache[parent]:
                    for suffix in ('_stageii.npz', '_poses.npz'):
                        if p.name.endswith(suffix) and normalized(p.name[:-len(suffix)]) == normalized(filename):
                            equivalent.append(p)
                if len(equivalent) == 1:
                    candidate = str(equivalent[0])
                    status = 'candidate_normalized_name_match'
                elif equivalent:
                    status = 'ambiguous_normalized_name'
        reference = None
        if dataset == 'SSM_synced':
            ref = SSM_REFERENCE / PurePosixPath(*path.parts[2:]).with_suffix('.npz')
            if ref.is_file():
                reference = str(ref)
        counters[dataset][status] += 1
        recordings[(dataset, status)].add(row['source_path'])
        records.append({'motion_id': mid, 'official_split': split, 'mirrored': mid.startswith('M'),
            'source_path': row['source_path'], 'index_start_20': row['start_frame'], 'index_end_20': row['end_frame'],
            'status': status, 'candidate_local_path': candidate, 'candidate_smplh_reference': reference})
    OUTPUT.mkdir(mode=0o700)
    with (OUTPUT / 'mapping.jsonl').open('x', encoding='utf8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = {'total_ids':len(records), 'index_rows':len(rows), 'per_dataset':dict(counters),
        'unique_source_paths': {d+'|'+s:len(v) for (d,s),v in recordings.items()},
        'input_sha256': {p.name:digest(p) for p in [PUBLIC/'index.csv', *(PUBLIC/'HumanML3D'/f'{s}.txt' for s in ('all','train','val','test'))]},
        'mapping_sha256':digest(OUTPUT/'mapping.jsonl'),
        'smplh_reference_ids':sum(1 for r in records if r['candidate_smplh_reference']),
        'limitations':['Only the eleven explicitly named AMASS roots and SSM reference were searched; other server locations may contain missing files.',
                       'Matches are inferred from aliases and file paths, not verified recording identity, hashes or timeline.',
                       'Available exports are not confirmed equivalent to original SMPL+H; FPS, trimming and frame correspondence remain unvalidated.',
                       'Mirrored IDs require transformed motion, not duplicate source file.',
                       'Path match is not proof of reconstructability, license, dynamics, or acceptance for training.']}
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
