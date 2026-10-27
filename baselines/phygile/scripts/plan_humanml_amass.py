"""Plan licensed AMASS acquisition and caption alignment; never fabricate motions.

Standard library only. Output is a plan, not a prepared training dataset.
Temporal rules follow HumanML3D raw_pose_processing.ipynb at the pinned revision.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

REVISION = '9176e8fb446b71c7d2a725eb5cf6fec1ae3b3c23'
TRIM_20 = {'Eyes_Japan_Dataset': 60, 'MPI_HDM05': 60,
           'TotalCapture': 20, 'MPI_Limits': 20, 'Transitions_mocap': 10}
DOWNLOAD_NAMES = {'BioMotionLab_NTroje': 'BMLrub', 'DFaust_67': 'DFaust',
                  'Eyes_Japan_Dataset': 'EyesJapanDataset',
                  'MPI_HDM05': 'HDM05', 'MPI_Limits': 'PosePrior',
                  'MPI_mosh': 'MoSh', 'SSM_synced': 'SSM',
                  'TCD_handMocap': 'TCDHands', 'Transitions_mocap': 'Transitions'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_parts(source):
    parts = PurePosixPath(source).parts
    if len(parts) < 4 or parts[0] != 'pose_data' or '..' in parts:
        raise ValueError('Unexpected HumanML source path: ' + source)
    return parts[1], str(PurePosixPath(*parts[2:]))


def parse_caption(line):
    fields = line.strip().split('#')
    if len(fields) != 4 or not fields[0].strip():
        raise ValueError('Malformed or empty caption')
    start, end = float(fields[2]), float(fields[3])
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError('Nonfinite time tag')
    if start == 0 and end == 0:
        return {'kind': 'whole', 'start_20': None, 'end_20': None}
    if start < 0 or end <= start:
        raise ValueError('Unordered time tag')
    # Match HumanML text loaders' integer 20-fps indexing, not rounded seconds.
    first, stop = int(start * 20), int(end * 20)
    if stop <= first:
        raise ValueError('Sub-frame caption interval')
    return {'kind': 'segment', 'start_20': first, 'end_20': stop}


def original_frame_indices(row, reference_fps, reference_num_frames, caption=None):
    """Exact reference-export sample indices, including Python's end=-1 rule.

    A matching SMPL-X export still needs its own validated time correspondence.
    These indices must not be applied unchanged to a differently sampled export.
    """
    if not math.isfinite(reference_fps) or reference_fps < 20:
        raise ValueError('Reference fps must be finite and >=20')
    if reference_num_frames < 2:
        raise ValueError('Reference frame count is required')
    dataset, _ = source_parts(row['source_path'])
    if dataset == 'humanact12':
        raise ValueError('HumanAct12 is a separate skeleton source, not AMASS')
    stride = int(reference_fps / 20)
    indices = list(range(0, reference_num_frames, stride))
    indices = indices[TRIM_20.get(dataset, 0):]
    start, stop = int(row['start_frame']), int(row['end_frame'])
    if start < 0 or stop < -1 or (stop >= 0 and stop <= start):
        raise ValueError('Invalid index.csv crop')
    if start >= len(indices) or (stop > len(indices)):
        raise ValueError('Index crop exceeds reference recording')
    indices = indices[start:stop]
    if caption is not None and caption['kind'] == 'segment':
        first, end = caption['start_20'], caption['end_20']
        if first < 0 or end > len(indices):
            raise ValueError('Caption exceeds indexed motion; refusing silent clamp')
        indices = indices[first:end]
    if len(indices) < 2:
        raise ValueError('Too few aligned frames')
    return indices


def make_plan(public, output, provenance):
    public, output = Path(public), Path(output)
    required = ['index.csv', 'HumanML3D/texts.zip'] + [
        'HumanML3D/' + split + '.txt' for split in ('all', 'train', 'val', 'test')]
    missing = [name for name in required if not (public / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing public metadata: ' + ', '.join(missing))
    provenance = Path(provenance)
    recorded = json.loads(provenance.read_text(encoding='utf-8'))
    if recorded['pinned_commit'] != REVISION:
        raise ValueError('Metadata provenance revision mismatch')
    hashes = {f['path']: f['sha256'] for f in recorded['files']}
    for name in required:
        if sha(public / name) != hashes[name]:
            raise ValueError('Public input differs from pinned acquisition: ' + name)
    rows = list(csv.DictReader((public / 'index.csv').open(encoding='utf-8', newline='')))
    index = {Path(row['new_name']).stem: row for row in rows}
    splits = {}
    for split in ('train', 'val', 'test'):
        for mid in (public / 'HumanML3D' / (split + '.txt')).read_text().splitlines():
            if mid in splits:
                raise ValueError('ID occurs in multiple official splits')
            splits[mid] = split
    all_ids = set((public / 'HumanML3D/all.txt').read_text().splitlines())
    if set(splits) != all_ids:
        raise ValueError('Official split union differs from all.txt')
    output.mkdir(parents=True, exist_ok=False)
    source_ids, source_splits = collections.defaultdict(list), collections.defaultdict(set)
    dataset_ids, counts, invalid = collections.Counter(), collections.Counter(), []
    seen_rows = set()
    with zipfile.ZipFile(public / 'HumanML3D/texts.zip') as z, (
            output / 'alignment_plan.jsonl').open('w', encoding='utf-8') as plan:
        if z.testzip() is not None:
            raise ValueError('Caption archive failed CRC')
        members = {PurePosixPath(n).stem: n for n in z.namelist() if n.endswith('.txt')}
        for mid, split in sorted(splits.items()):
            canonical = mid[1:] if mid.startswith('M') else mid
            row = index[canonical]
            dataset, recording = source_parts(row['source_path'])
            source = row['source_path']
            source_ids[source].append(mid)
            source_splits[source].add(split)
            dataset_ids[dataset] += 1
            seen_rows.add(canonical)
            accepted = []
            for line_no, line in enumerate(z.read(members[mid]).decode('utf-8').splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = parse_caption(line)
                    accepted.append({'line': line_no, **item})
                    counts[item['kind'] + '_caption_lines'] += 1
                except (ValueError, OverflowError) as error:
                    invalid.append({'motion_id': mid, 'line': line_no, 'reason': str(error)})
            record = {'motion_id': mid, 'official_split': split, 'source_group': source,
                      'dataset': dataset, 'recording': recording,
                      'index_start_20': int(row['start_frame']),
                      'index_end_20': int(row['end_frame']),
                      'source_prefix_trim_20': TRIM_20.get(dataset, 0),
                      'mirrored': mid.startswith('M'), 'captions': accepted,
                      'status': 'pending_licensed_raw_motion_and_export_timeline',
                      'amass_source': dataset != 'humanact12'}
            plan.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    with (output / 'source_requirements.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'amass_download_name', 'humanml_source_path',
                         'expected_original_smplh_file', 'paired_ids', 'official_splits',
                         'prefix_trim_20', 'source_kind'])
        for source, ids in sorted(source_ids.items()):
            dataset, recording = source_parts(source)
            writer.writerow([dataset, DOWNLOAD_NAMES.get(dataset, dataset), source,
                             str(PurePosixPath(dataset) / PurePosixPath(recording).with_suffix('.npz')),
                             len(ids), '|'.join(sorted(source_splits[source])),
                             TRIM_20.get(dataset, 0),
                             'separate_HumanAct12' if dataset == 'humanact12' else 'AMASS'])
    intersections = {}
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        intersections[left + '_' + right] = sum(
            {left, right}.issubset(parts) for parts in source_splits.values())
    summary = {'status': 'PLAN_ONLY_AWAITING_ACQUIRED_INPUTS', 'date': '2026-10-21',
               'humanml_revision': REVISION, 'official_motion_ids': len(splits),
               'indexed_original_clips_in_official_splits': len(seen_rows),
               'unique_recording_sources': len(source_ids),
               'datasets': [{'humanml_name': name, 'download_name': DOWNLOAD_NAMES.get(name, name),
                             'motion_ids': dataset_ids[name],
                             'unique_sources': sum(source_parts(s)[0] == name for s in source_ids),
                             'amass_source': name != 'humanact12'} for name in sorted(dataset_ids)],
               'caption_counts': dict(counts), 'rejected_time_or_format_lines': len(invalid),
               'original_recording_overlap_across_official_splits': intersections,
               'raw_amass_files_acquired_this_run': 0, 'body_models_acquired_this_run': 0,
               'g1_paired_sequences_prepared': 0, 'generator_training_started': False,
               'input_provenance_sha256': sha(provenance),
               'source_files': {name: sha(public / name) for name in required},
               'outputs': {name: sha(output / name) for name in
                           ('alignment_plan.jsonl', 'source_requirements.csv')},
               'limitations': ['No raw-motion alignment, GMR retargeting or 262D encoding executed.',
                               'SMPL+H reference indices need validated matching SMPL-X export times.',
                               'Official ID splits are not recording-disjoint; training protocol unresolved.',
                               'Mirrored text IDs require actual anatomical mirroring of the motion.']}
    (output / 'rejected_caption_lines.json').write_text(
        json.dumps(invalid, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('datasets', 'source_files', 'outputs')},
                     ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, default=Path(__file__).resolve().parents[1] /
                        'results/paired_data_20261018/public_metadata_manifest.json')
    args = parser.parse_args()
    make_plan(args.public, args.output, args.provenance)
