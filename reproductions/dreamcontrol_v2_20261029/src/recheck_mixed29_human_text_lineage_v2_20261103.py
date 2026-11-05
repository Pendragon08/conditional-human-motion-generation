"""Recheck source-sequence/text lineage without relabeling or filtering motions.

GRAB identity uses previously verified publisher metadata, including the raw
human member SHA256. Omni provides paired human_joints arrays plus the OMOMO
sequence key, without a raw-human checksum. Report those evidence levels.
"""
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D = R / 'mixed29_human_text_lineage_recheck_20261103'
P = R / 'grab_omni_29dof_weak_baseline_v1_20261103'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def unique(rows, field):
    result = {row[field]: row for row in rows}
    require(len(result) == len(rows), 'duplicate ' + field)
    return result


def main():
    output = D / 'report_v2.json'
    require(not output.exists(), 'refusing to overwrite audit')
    paths = {
        'mixed_manifest': P / 'manifest.json',
        'raw_grab_registry': R / 'grab_source_registry_20261030/source_registry.jsonl',
        'motionhub_caption_snapshot': D / 'motionhub_captions.jsonl',
        'grab_candidate_pairs': R / 'grab_dex_candidate_pairs_20261103.jsonl',
        'omomo_candidate_pairs': R / 'omomo_group_split_20261101/candidates.jsonl',
        'omni_archive': R / 'data/OmniRetarget_Dataset/robot-object.zip',
        'omomo_text_archive': R / 'omomo_text_pairing_20261101/omomo_text_anno.zip',
    }
    expected = {
        'mixed_manifest': '5fc1a0beb071e7a65b68e689a57ed0ba09b633a83d5ec5fd261b1620115bb1cb',
        'raw_grab_registry': '27d74b72fec830afa01cc05f788d921ef3d42fc1115aad08565d26604cdec8e4',
        'motionhub_caption_snapshot': '2714374c8f91b88e1344ab3c4be8345eaea4cd076e1e8f5f0e8bb0bd38f58c25',
        'grab_candidate_pairs': 'ba68debdf5061aaf7f3b5fa3cf8ab64c4b1266f439bca187cbbce5b13f849104',
        'omomo_candidate_pairs': '905af6eefd88e3c06cd5ff47877cade07a9a379c1f79d46a71d227e911b31085',
        'omni_archive': '93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3',
        'omomo_text_archive': '2d11fb37fef294b7f6816c74740694d12f28d62e0f638e348c678dc51ee1ad7c',
    }
    observed = {name: sha(path) for name, path in paths.items()}
    require(observed == expected, 'pinned input changed')
    rows = json.loads(paths['mixed_manifest'].read_text())['records']
    require(len(unique(rows, 'sample_id')) == 3161, 'mixed coverage')
    registry = {x['member'][:-4]: x for x in jsonl(paths['raw_grab_registry'])}
    captions = unique(jsonl(paths['motionhub_caption_snapshot']), 'id')
    pairs = unique(jsonl(paths['grab_candidate_pairs']), 'id')
    omni_pairs = unique(jsonl(paths['omomo_candidate_pairs']), 'motion_member')
    require(len(registry) == len(captions) == len(pairs) == 1335, 'GRAB source coverage')

    logs = {}
    evidence = {}
    for batch in ('batch10', 'batch100', 'batch1224'):
        folder = R / ('dexweave_stream_robot_' + batch + '_20261103')
        rp = folder / 'records.jsonl'
        audit = json.loads((folder / 'audit.json').read_text())
        require(sha(rp) == audit['records_sha256'], 'GRAB producer log changed')
        require(audit['samples'] == audit['passed'], 'GRAB producer audit incomplete')
        evidence[batch] = {'records_sha256': sha(rp), 'audit_sha256': sha(folder / 'audit.json')}
        for record in jsonl(rp):
            require(record['id'] not in logs, 'duplicate GRAB producer source')
            logs[record['id']] = (folder, record)
    pilot_dir = R / 'dexweave_grab_pilot_20261103'
    pilot = json.loads((pilot_dir / 'preview_report.json').read_text())
    evidence['pilot_report_sha256'] = sha(pilot_dir / 'preview_report.json')
    require(len(logs) == 1334 and pilot['id'] not in logs, 'GRAB producer coverage')

    details, errors = [], []
    counts = Counter()
    crop = Counter()
    grab_ids = set()
    omni_keys = set()
    omni_shapes = {}
    omni_human = {}
    omni_npz_keysets = Counter()
    fps = Counter()
    with zipfile.ZipFile(paths['omni_archive']) as robots, zipfile.ZipFile(paths['omomo_text_archive']) as texts:
        for row in rows:
            try:
                artifact = R / row['state_path']
                require(sha(artifact) == row['state_sha256'], 'current artifact checksum')
                if row['dataset_source'] == 'DexWeave/GRAB':
                    sid = row['source_id'].removeprefix('GRAB_sequence:')
                    require(sid not in grab_ids, 'duplicate GRAB final source')
                    grab_ids.add(sid)
                    human, text, pair = registry[sid], captions[sid], pairs[sid]
                    require(row['sample_id'] == 'dex29_' + sid.replace('/', '_'), 'GRAB sequence ID')
                    require(pair['robot_relative_path'] == 'data/GRAB/' + sid + '/' + sid.split('/')[-1] + '_opt.npz', 'GRAB robot path identity')
                    require(pair['text_relative_path'] == 'GRAB/hierarchical_caption/' + sid + '.json', 'GRAB text path identity')
                    require(human['member_sha256'] == text['source_sha256'] == pair['source_member_sha256'] == row['source_sha256'], 'raw human identity hash')
                    require(text['text_sha256'] == pair['text_sha256'] == row['annotation_sha256'], 'MotionHub annotation identity')
                    source_text = text['annotation']['macro']
                    require(source_text == pair['captions'] == row['candidate_captions'], 'source captions differ')
                    require(row['caption'] == source_text[0], 'primary source caption differs')
                    require(row['provenance_keys']['source_member_sha256'] == [human['member_sha256']], 'human provenance keys')
                    detail = {'sample_id': row['sample_id'], 'human_sequence': sid,
                              'human_sha256': human['member_sha256'], 'text_sha256': text['text_sha256'],
                              'unchanged_captions': len(source_text), 'publisher_source_identity_evidence': 'raw_human_sha256_in_verified_robot_metadata'}
                    if sid == pilot['id']:
                        require(row['source_robot_sha256'] == pilot['motion_sha256'], 'pilot robot identity')
                        raw_path = pilot_dir / 'wineglass_pass_1_opt.npz'
                        require(sha(raw_path) == row['source_robot_sha256'], 'pilot raw robot checksum')
                        with np.load(raw_path, allow_pickle=False) as z:
                            require(z['source_seq_name'].item() == sid and z['source_motion_sha256'].item() == human['member_sha256'], 'pilot raw human metadata')
                            require(int(z['source_frame_count']) == human['frames'], 'pilot source frame count')
                            indices = np.asarray(z['retime_source_indices'])
                            require(len(indices) > 1 and np.all(np.diff(indices) >= 0) and indices[0] >= 0 and indices[-1] <= human['frames'] - 1, 'pilot source index range')
                            first, last = float(indices[0]), float(indices[-1])
                        require(row['state_sha256'] == pilot['output_sha256'], 'pilot derived artifact')
                    else:
                        folder, log = logs[sid]
                        require(log['status'] == 'technical_preview', 'failed robot import')
                        require(row['source_robot_sha256'] == log['input_sha256'], 'published robot identity')
                        require(log['source_member_sha256'] == human['member_sha256'] and log['source_frame_count'] == human['frames'], 'recorded robot to raw human binding')
                        require(log['text_sha256'] == text['text_sha256'] and log['captions'] == source_text, 'recorded caption binding')
                        require(artifact == folder / log['artifact'] and row['state_sha256'] == log['artifact_sha256'], 'derived artifact identity')
                        first, last = log['source_index_first'], log['source_index_last']
                        require(0 <= first <= last <= human['frames'] - 1, 'recorded source index range')
                    detail.update(source_index_first=first, source_index_last=last, raw_human_frame_count=human['frames'])
                    counts['grab_trajectories'] += 1
                    counts['grab_unchanged_captions'] += len(source_text)
                else:
                    member = row['source_member']
                    stem = Path(member).stem
                    match = re.fullmatch(r'(sub\d+_.+_\d+)_(original|trans_\d+|rot_\d+)', stem)
                    require(match is not None, 'unrecognized Omni source suffix')
                    key, variant = match.groups()
                    original_member = 'robot-object/' + key + '_original.npz'
                    require(original_member in robots.namelist(), 'missing original source-group robot')
                    annotation_member = 'omomo_text_anno_json_data/' + key + '.json'
                    ann = texts.read(annotation_member)
                    annotation = json.loads(ann)
                    require(set(annotation) == {key}, 'annotation has unexpected source key')
                    source_text = annotation[key]
                    require(row['source_id'] == 'OMOMO_nominal_action:' + key, 'Omni nominal human sequence ID')
                    require(row['caption_source'] == 'OMOMO author archive: ' + annotation_member, 'Omni text member association')
                    require(hashlib.sha256(ann).hexdigest() == row['annotation_sha256'], 'original OMOMO text checksum')
                    require(row['caption'] == source_text and row.get('candidate_captions', [row['caption']]) == [source_text], 'original OMOMO wording differs')
                    candidate = omni_pairs[member]
                    require(candidate['nominal_action_key'] == key and candidate['annotation_member'] == annotation_member and candidate['variant'] == variant, 'Omni candidate source association')
                    require(candidate['caption'] == source_text and candidate['annotation_sha256'] == row['annotation_sha256'], 'Omni candidate caption differs')
                    raw = robots.read(member)
                    require(hashlib.sha256(raw).hexdigest() == row['source_sha256'] == candidate['motion_member_sha256'], 'released Omni robot checksum')
                    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                        keys = sorted(z.files)
                        shape = z['qpos'].shape
                        source_fps = float(z['fps'])
                        h = np.asarray(z['human_joints'])
                        require(h.shape == (shape[0], 52, 3) and np.isfinite(h).all(), 'paired human/robot frame layout')
                        human_sha = hashlib.sha256(h.tobytes(order='C')).hexdigest()
                    require(shape[1] == 43 and source_fps == 30., 'Omni source layout/time declaration')
                    omni_shapes[member] = (shape, source_fps)
                    omni_human[member] = (h.shape, str(h.dtype), human_sha)
                    omni_npz_keysets['|'.join(keys)] += 1
                    omni_keys.add(key)
                    counts['omni_originals' if variant == 'original' else 'omni_variants'] += 1
                    counts['omni_trajectories'] += 1
                    counts['omni_unchanged_captions'] += 1
                    counts['omni_files_with_paired_human_joints'] += 1
                    if not any('sha' in k.lower() or 'hash' in k.lower() for k in keys):
                        counts['omni_files_without_raw_human_checksum'] += 1
                    detail = {'sample_id': row['sample_id'], 'human_sequence_key': key,
                              'robot_source_member': member, 'original_robot_member': original_member,
                              'annotation_member': annotation_member, 'text_sha256': row['annotation_sha256'],
                              'variant': variant, 'unchanged_captions': 1,
                              'paired_human_joints_sha256': human_sha,
                              'paired_human_joints_shape': list(h.shape),
                              'source_identity_evidence': 'publisher_paired_human_joints_and_OMOMO_sequence_key_no_raw_human_checksum'}
                with np.load(artifact, allow_pickle=False) as z:
                    times = np.asarray(z['sample_times'])
                require(row['fps'] == 20. and np.allclose(times, np.arange(len(times)) / 20., atol=1e-10, rtol=0), 'current output time grid')
                require(np.allclose(row['nominal_sample_time_s'], [times[0], times[-1]], atol=1e-10, rtol=0), 'manifest time span differs')
                fps['20Hz'] += 1
                detail.update(cropped_10s=row['cropped_for_max_10s'], output_samples=len(times), nominal_sample_span_s=[float(times[0]), float(times[-1])])
                if row['cropped_for_max_10s']:
                    crop[row['dataset_source']] += 1
                details.append(detail)
            except Exception as exc:
                errors.append({'sample_id': row.get('sample_id'), 'error': type(exc).__name__ + ': ' + str(exc)})
    require(grab_ids == set(registry) == set(captions) == set(pairs), 'GRAB set mismatch')
    variant_length_changes = []
    different_human_variants = []
    identical_human_variants = 0
    for member, (shape, f) in omni_shapes.items():
        key = omni_pairs[member]['nominal_action_key']
        original = 'robot-object/' + key + '_original.npz'
        if original in omni_shapes and omni_shapes[original] != (shape, f):
            variant_length_changes.append(member)
        if omni_pairs[member]['variant'] != 'original':
            if omni_human[member] == omni_human[original]:
                identical_human_variants += 1
            else:
                different_human_variants.append(member)
    details_path = D / 'records_v2.jsonl'
    with details_path.open('x', encoding='utf8') as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    report = {
        'scope': 'source_sequence_and_verbatim_published_text_recheck_no_semantic_or_physical_screen',
        'total_records': len(rows), 'passed_source_associations': len(details), 'errors': errors,
        'counts': dict(counts), 'omomo_unique_human_sequence_keys': len(omni_keys),
        'cropped_10s': dict(crop), 'output_time_grid_counts': dict(fps),
        'omni_variants_with_different_nominal_length_from_original': variant_length_changes,
        'omni_variants_with_identical_paired_human_joints_to_original': identical_human_variants,
        'omni_variants_with_different_paired_human_joints_to_original': different_human_variants,
        'omni_source_npz_keysets': dict(omni_npz_keysets),
        'input_sha256': observed, 'producer_evidence': evidence, 'records_sha256': sha(details_path),
        'audit_script_sha256': sha(Path(__file__)),
        'supersedes': 'report.json preliminary inference that Omni NPZ had only fps/qpos; actual files also contain human_joints, now checked in full',
        'dataset_manifest_changed': sha(paths['mixed_manifest']) != expected['mixed_manifest'],
        'training_started': False,
        'interpretation': [
            'All final source/annotation bindings are compared against the preserved raw-human registry, downloaded publisher-caption snapshot, and/or original OMOMO annotation zip.',
            'GRAB uses MotionHub annotations of the same raw GRAB sequence; these are not claimed to be GRAB-official prose.',
            'For streamed DexWeave references the embedded human checksum and retiming contract were verified at ingestion; this recheck binds those hash-pinned producer records rather than redownloading 1335 robot files.',
            'Omni published files include qpos, human_joints and fps: all paired human/robot frame counts and finite human arrays are checked, and variant human arrays are compared byte-for-byte to their source-group original.',
            'The 312 OMOMO sequence identifiers link those paired human arrays to original annotations; the release embeds no raw-OMOMO checksum, so raw-recording identity is not independently hash-certified.',
            'The 1514 Omni variants inherit their source-sequence captions. Matching wording does not prove every transformed/cropped robot clip realizes every sentence event.',
            '20Hz resampling preserves source association; the 513 truncated clips retain full-source captions, so full temporal/event equivalence is not asserted.',
            'This audit does not change labels, discard data, restart retargeting, add a physical-action filter, or invalidate the completed user-requested dataset-building scope.',
        ],
    }
    with output.open('x', encoding='utf8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('producer_evidence', 'input_sha256', 'interpretation')}, ensure_ascii=False), flush=True)
    require(not errors and len(details) == 3161, 'source pairing errors found; see report')


if __name__ == '__main__':
    main()
