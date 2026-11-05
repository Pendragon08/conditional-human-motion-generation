"""Build an explicitly non-official source-exclusive evaluation view.

Official ID files and motion data are untouched. In each exact source-path group,
retain IDs only from the highest-priority original split (test > val > train).
All discarded IDs remain in the manifest with explicit reasons.
"""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OUT=Path(__file__).resolve().parent
MAPPING=ROOT/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
EXPECTED='358875ac6aca9e976086778fe317f27f64134e1fafaff51581817c09f22da778'
PRIORITY={'train':0,'val':1,'test':2}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    if sha(MAPPING)!=EXPECTED:raise ValueError('mapping changed')
    data=[json.loads(s) for s in MAPPING.read_text().splitlines()]
    if len(data)!=29228 or len({r['motion_id'] for r in data})!=29228:raise ValueError('wrong input coverage')
    groups=defaultdict(list)
    for r in data:
        if r['official_split'] not in PRIORITY or not r['source_path'].startswith('./pose_data/'):
            raise ValueError('bad input row')
        groups[r['source_path']].append(r)
    manifest=[];retained=Counter();excluded=Counter();retained_paths=defaultdict(set)
    for path,rows in groups.items():
        highest=max((r['official_split'] for r in rows),key=lambda s:PRIORITY[s])
        for r in rows:
            keep=r['official_split']==highest
            (retained if keep else excluded)[r['official_split']]+=1
            if keep:retained_paths[r['official_split']].add(path)
            manifest.append({'motion_id':r['motion_id'],'official_split':r['official_split'],
                             'exact_source_path':path,'mirrored':r['mirrored'],'mapping_status':r['status'],
                             'view_split':highest if keep else None,'view_disposition':'retained' if keep else 'excluded_source_path_in_higher_priority_split',
                             'training_ready':False})
    if any(retained_paths[a]&retained_paths[b] for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('view contains cross-split exact paths')
    if retained['test']!=4384 or sum(retained.values())+sum(excluded.values())!=29228:raise ValueError('unexpected coverage')
    path=OUT/'view.jsonl'
    with path.open('x',encoding='utf8',newline='\n') as f:
        for r in sorted(manifest,key=lambda x:x['motion_id']):f.write(json.dumps(r,sort_keys=True)+'\n')
    report={'mapping_sha256':EXPECTED,'script_sha256':sha(__file__),'view_sha256':sha(path),
            'policy':'Keep official test; then val; then train, excluding lower-priority IDs whose exact indexed source_path appears in higher-priority splits.',
            'official_counts':dict(Counter(r['official_split'] for r in data)),
            'retained_counts':dict(retained),'excluded_counts':dict(excluded),
            'retained_exact_path_counts':{s:len(retained_paths[s]) for s in PRIORITY},
            'limitations':'Supplementary non-official manifest only. Exact index path is not verified recording identity; aliases or related recordings can still leak. Exclusions may remove valid disjoint clips. No source, official split or training data changed; FPS, license, quality and train readiness remain unverified.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps(report,sort_keys=True),flush=True)

if __name__=='__main__':main()
