"""Deterministic source-grouped inventory; never select on execution quality."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX_URL = 'https://raw.githubusercontent.com/EricGuo5513/HumanML3D/main/index.csv'


def stable_key(value):
    return hashlib.sha256(('physics-guided-v1:' + value).encode()).hexdigest()


def prepare(output, snapshot, config):
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / 'humanml_source_index.csv'
    if not index_path.exists():
        content = urllib.request.urlopen(INDEX_URL, timeout=30).read()
        index_path.write_bytes(content)
    index = {Path(r['new_name']).stem: r for r in csv.DictReader(io.StringIO(index_path.read_text()))}
    data = snapshot / 'data/dataset/HumanML3D'
    cached = {}
    for split in ('train','test'):
        path = snapshot / 'data/humanml3d' / ('t2m_'+split+'.npy')
        package=np.load(path,allow_pickle=True).item()
        for key, value in package['data_dict'].items():
            if re.fullmatch(r'[0-9]{6}',key):
                cached[key]=(value,path,split)
    patterns = {
        'jump': r'\b(jump|jumps|jumping|hop|hops|hopping|leap|leaps|leaping)\b',
        'dance': r'\b(dance|dances|dancing|ballet)\b'
    }
    groups = {f: {} for f in patterns}
    inventory = []
    for clip_id in sorted(cached):
        if clip_id not in index:
            continue
        value, cache_path, official_split = cached[clip_id]
        motion = value['motion']
        if len(motion) < config['context_frames_20hz'] + config['prediction_frames_20hz']:
            continue
        source = index[clip_id]['source_path']
        matched = []
        for family, pattern in patterns.items():
            for text in value['text']:
                caption, tokens = text['caption'], '_'.join(text['tokens'])
                # Only full-clip captions; cropped captions are not silently
                # assigned to a different temporal interval.
                if re.search(pattern, caption, re.I):
                    matched.append((family, caption, tokens))
                    break
        if len(matched) != 1:
            continue  # Ambiguous jump+dance captions are excluded before runs.
        family, caption, tokens = matched[0]
        row = dict(clip_id=clip_id, source_id=source, family=family, caption=caption,
                   tokens=tokens, start_frame=0, available_frames=len(motion),
                   cache_file=str(cache_path.resolve()), cache_key=clip_id,
                   official_split=official_split, official_source=index[clip_id])
        groups[family].setdefault(source, []).append(row)
        inventory.append(row)
    ambiguous_sources=set(groups['jump']) & set(groups['dance'])
    for g in groups.values():
        for s in ambiguous_sources: g.pop(s,None)
    counts = {f: len(g) for f, g in groups.items()}
    required = sum(config['source_counts_per_family'].values())
    (output / 'inventory.json').write_text(json.dumps(dict(
        counts=counts, required_per_family=required, eligible_clips=len(inventory),
        source_index_url=INDEX_URL, source_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        eligibility='full caption, unmirrored, >=80 frames, exactly one task family',
        grouping='original source_path, including all temporal cuts and mirrors',
        split_note='new experiment source-group split; not a claim of pretraining-data independence'
    ), indent=2), encoding='utf8')
    if any(v < required for v in counts.values()):
        raise RuntimeError('Insufficient independent original sources: ' + repr(counts))
    rows = []
    for family in config['families']:
        source_ids = sorted(groups[family], key=stable_key)
        cursor = 0
        for split in ('test', 'val', 'train'):
            for source in source_ids[cursor:cursor + config['source_counts_per_family'][split]]:
                row = sorted(groups[family][source], key=lambda r: stable_key(r['clip_id']))[0].copy()
                row['split'] = split
                row['goal_joint'] = 'pelvis' if family == 'jump' else 'right_wrist'
                rows.append(row)
            cursor += config['source_counts_per_family'][split]
    assert len({r['source_id'] for r in rows}) == len(rows)
    manifest = dict(snapshot=str(snapshot), rows=rows, mean=str(data/'Mean.npy'), std=str(data/'Std.npy'),
                    config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    path = output / 'source_manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Cannot replace a frozen source manifest'
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('PHYSICS_GUIDED_DATA_PREPARED', counts, len(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--config', default=str(ROOT/'configs/physics_guided_20261011.json'))
    a = p.parse_args()
    prepare(Path(a.output), Path(a.snapshot), json.loads(Path(a.config).read_text()))
