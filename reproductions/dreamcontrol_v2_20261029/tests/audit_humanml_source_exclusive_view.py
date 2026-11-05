"""Independently verify view coverage, exclusions, and exact-path isolation."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
MAPPING=ROOT/'humanml_raw_mapping_v2_20261030/mapping.jsonl'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    report=json.loads((OUT/'report.json').read_text());view=OUT/'view.jsonl'
    if sha(MAPPING)!=report['mapping_sha256'] or sha(view)!=report['view_sha256'] or sha(OUT/'build_humanml_source_exclusive_view.py')!=report['script_sha256']:
        raise ValueError('input or output changed')
    original={x['motion_id']:x for x in map(json.loads,MAPPING.read_text().splitlines())}
    observed=[json.loads(x) for x in view.read_text().splitlines()]
    if len(original)!=29228 or len(observed)!=29228 or len({x['motion_id'] for x in observed})!=29228 or set(original)!={x['motion_id'] for x in observed}:
        raise ValueError('IDs do not match exactly')
    by_path=defaultdict(list)
    for r in observed:
        raw=original[r['motion_id']]
        if (r['official_split']!=raw['official_split'] or r['exact_source_path']!=raw['source_path'] or r['mirrored']!=raw['mirrored'] or
            r['mapping_status']!=raw['status'] or r['training_ready'] is not False):raise ValueError('row provenance mismatch')
        by_path[r['exact_source_path']].append(r)
    accepted=Counter();rejected=Counter();cross=defaultdict(set)
    for path,rows in by_path.items():
        splits={x['official_split'] for x in rows}
        winner='test' if 'test' in splits else ('val' if 'val' in splits else 'train')
        for r in rows:
            should_keep=r['official_split']==winner
            if should_keep:
                if r['view_disposition']!='retained' or r['view_split']!=r['official_split']:raise ValueError('incorrect retained ID')
                accepted[r['official_split']]+=1;cross[path].add(r['view_split'])
            else:
                if r['view_disposition']!='excluded_source_path_in_higher_priority_split' or r['view_split'] is not None:raise ValueError('incorrect exclusion')
                rejected[r['official_split']]+=1
    if any(len(v)>1 for v in cross.values()) or accepted['test']!=4384 or dict(accepted)!=report['retained_counts'] or dict(rejected)!=report['excluded_counts']:
        raise ValueError('split invariant mismatch')
    out={'script_sha256':sha(__file__),'view_sha256':sha(view),'report_sha256':sha(OUT/'report.json'),
         'ids_verified':len(observed),'unique_index_paths':len(by_path),'retained_counts':dict(accepted),
         'excluded_counts':dict(rejected),'retained_cross_split_exact_path_count':sum(len(x)>1 for x in cross.values()),
         'limitations':'Checks exact input paths and policy, not capture-level identity, alias equivalence, official evaluation, licenses, real FPS or training quality.'}
    with (OUT/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps(out,sort_keys=True),flush=True)
if __name__=='__main__':main()
