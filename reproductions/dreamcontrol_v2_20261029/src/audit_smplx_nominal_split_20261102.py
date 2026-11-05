"""Recompute same-source nominal overlap by original paths and direct intersections."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mapping', required=True, type=Path)
    p.add_argument('--inventory', required=True, type=Path)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    paths = [a.mapping, a.inventory/'records.jsonl', a.report]
    hashes = {str(path): sha(path) for path in paths}
    ids = {row['motion_id'] for line in paths[1].read_text(encoding='utf8').splitlines()
           for row in json.loads(line)['nominal_rows']}
    groups = defaultdict(list)
    for line in a.mapping.read_text(encoding='utf8').splitlines():
        r = json.loads(line)
        if r['motion_id'] in ids:
            groups[r['source_path']].append(r)
    expected = set()
    for rows in groups.values():
        for i, x in enumerate(rows):
            for y in rows[i+1:]:
                if x['official_split'] == y['official_split'] or x['mirrored'] != y['mirrored']:
                    continue
                start = max(int(x['index_start_20']), int(y['index_start_20']))
                end = min(int(x['index_end_20']), int(y['index_end_20']))
                if start < end:
                    expected.add((tuple(sorted((x['motion_id'], y['motion_id']))), start, end, x['mirrored']))
    report = json.loads(a.report.read_text(encoding='utf8'))
    actual = [(tuple(sorted(pair['ids'])), *pair['nominal_shared_interval'], pair['mirrored'])
              for group in report['groups'] for pair in group['nominal_same_transform_overlaps']]
    if len(actual) != len(set(actual)) or set(actual) != expected or len(actual) != report['same_transform_nominal_overlap_pairs']:
        raise ValueError('overlap pairs differ from direct mapping')
    if any(sha(path) != hashes[str(path)] for path in paths):
        raise RuntimeError('audit inputs changed')
    result = {'matched_ids': len(ids), 'nominal_overlap_pairs_verified': len(expected), 'input_sha256': hashes,
              'script_sha256': sha(Path(__file__)), 'training_ready': False,
              'limitations': ['Only nominal labels and ranges are checked; no actual motion bytes or clock identity were compared.']}
    with a.output.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'matched_ids': len(ids), 'nominal_overlap_pairs_verified': len(expected)}))


if __name__ == '__main__':
    main()
