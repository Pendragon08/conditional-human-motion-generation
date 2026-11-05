"""Independently audit source split groups and their nominal index intersections."""
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
RUN = ROOT / 'humanml_source_split_audit_20261030'
MAPPING = ROOT / 'humanml_raw_mapping_v2_20261030/mapping.jsonl'
EXPECTED = '358875ac6aca9e976086778fe317f27f64134e1fafaff51581817c09f22da778'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024), b''): h.update(x)
    return h.hexdigest()

def lines(p):
    return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]

def main():
    original = lines(MAPPING)
    report = json.loads((RUN / 'report.json').read_text())
    if sha(MAPPING) != EXPECTED or report['mapping_sha256'] != EXPECTED or sha(RUN / 'cross_split_sources.jsonl') != report['cross_split_sources_sha256']:
        raise ValueError('input changed')
    allgroups = defaultdict(list)
    for row in original: allgroups[row['source_path']].append(row)
    cross = {p:rs for p,rs in allgroups.items() if len({r['official_split'] for r in rs}) > 1}
    output = lines(RUN / 'cross_split_sources.jsonl')
    if len(output) != len(cross) or len({x['source_path'] for x in output}) != len(cross):
        raise ValueError('group coverage differs')
    intersect_groups, unknown_groups = [], []
    intersect_ids = set()
    split_pairs, dataset_counts = Counter(), Counter()
    for g in output:
        src = cross[g['source_path']]
        expected_members = {(r['motion_id'],r['official_split'],r['mirrored'],str(r['index_start_20']),str(r['index_end_20']),r['status']) for r in src}
        actual_members = {(r['motion_id'],r['split'],r['mirrored'],str(r['index_start_20']),str(r['index_end_20']),r['mapping_status']) for r in g['members']}
        if expected_members != actual_members or len(actual_members) != len(g['members']): raise ValueError('member mismatch')
        overlaps, unknown = [], False
        for a,b in combinations(g['members'], 2):
            if a['split'] == b['split']: continue
            try: a0,a1,b0,b1 = map(int, (a['index_start_20'],a['index_end_20'],b['index_start_20'],b['index_end_20']))
            except ValueError: unknown = True; continue
            if a0 < 0 or b0 < 0 or a1 < 0 or b1 < 0 or a1 <= a0 or b1 <= b0:
                unknown = True; continue
            start,end = max(a0,b0),min(a1,b1)
            if start < end:
                overlaps.append({'ids':sorted((a['motion_id'],b['motion_id'])), 'splits':sorted((a['split'],b['split'])), 'nominal_index_overlap':[start,end]})
                intersect_ids.update((a['motion_id'],b['motion_id']))
                split_pairs['+'.join(sorted((a['split'],b['split'])))] += 1
        if unknown: unknown_groups.append(g['source_path'])
        if overlaps:
            dataset_counts[g['dataset']] += 1
            intersect_groups.append({'source_path':g['source_path'], 'pairs':overlaps})
    output_path = RUN / 'nominal_cross_split_intersections.jsonl'
    with output_path.open('x', encoding='utf8', newline='\n') as f:
        for item in intersect_groups: f.write(json.dumps(item, sort_keys=True) + '\n')
    result = {'mapping_sha256':EXPECTED, 'input_report_sha256':sha(RUN/'report.json'), 'script_sha256':sha(Path(__file__)),
              'intersection_output_sha256':sha(output_path), 'independently_verified_cross_split_paths':len(cross),
              'paths_with_strict_nominal_index_intersections':len(intersect_groups), 'ids_in_nominal_intersections':len(intersect_ids),
              'paths_with_unknown_or_invalid_index_pair':len(unknown_groups), 'pair_count_by_split':dict(split_pairs),
              'paths_with_intersections_by_dataset':dict(dataset_counts),
              'assumptions':'For exploration only, index pairs treated as half-open intervals [start,end); negative or invalid indices are unknown. Source index meaning, actual FPS, decoding and mirrored transformations remain unverified. Counting IDs includes mirrored versions. Potential leakage is not a train-ready split or demonstrated physical-time overlap.'}
    with (RUN/'independent_interval_audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,sort_keys=True)
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__ == '__main__': main()
