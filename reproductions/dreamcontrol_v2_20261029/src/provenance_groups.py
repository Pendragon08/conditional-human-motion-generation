"""Audit explicit provenance relations across proposed dataset splits.

Each record supplies one or more namespaced group keys. Records sharing any
key are connected transitively. The caller must derive keys from evidence;
this tool cannot discover unrecorded aliases, people or related recordings.
"""
from collections import defaultdict


def audit_provenance(records):
    if not isinstance(records, list) or not records:
        raise ValueError('nonempty records required')
    parents = list(range(len(records)))
    seen_ids, first = set(), {}
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(i, j):
        a, b = root(i), root(j)
        if a != b:
            parents[max(a, b)] = min(a, b)
    for i, rec in enumerate(records):
        sample = rec.get('sample_id')
        split = rec.get('split')
        keys = rec.get('provenance_keys')
        if not isinstance(sample, str) or not sample.strip() or sample in seen_ids:
            raise ValueError('missing or duplicate sample_id')
        if split not in ('train', 'val', 'test'):
            raise ValueError('invalid split')
        if not isinstance(keys, dict) or not keys:
            raise ValueError(f'missing explicit provenance keys for {sample}')
        seen_ids.add(sample)
        for namespace, values in keys.items():
            if not isinstance(namespace, str) or not namespace.strip():
                raise ValueError('invalid provenance namespace')
            if not isinstance(values, list) or not values:
                raise ValueError('each provenance namespace requires a nonempty list')
            if any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError('invalid provenance value')
            for value in values:
                key = (namespace, value)
                if key in first: union(i, first[key])
                else: first[key] = i
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[root(i)].append({'sample_id': rec['sample_id'], 'split': rec['split']})
    components = [sorted(group, key=lambda r: r['sample_id']) for group in groups.values()]
    components.sort(key=lambda g: g[0]['sample_id'])
    crossings = [g for g in components if len({r['split'] for r in g}) > 1]
    return {'components': components, 'component_count': len(components),
            'cross_split_components': crossings, 'source_groups_disjoint': not crossings,
            'limitations': ['Depends on completeness and correctness of caller-supplied provenance keys.',
                            'Does not verify license, physical quality, caption semantics or identity.']}
