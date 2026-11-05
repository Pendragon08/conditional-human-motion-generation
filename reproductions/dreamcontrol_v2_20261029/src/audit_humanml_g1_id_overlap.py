"""Compare identifier sets, without claiming source or movement equivalence."""
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MAP=ROOT/'humanml_raw_mapping_v2_20261030/mapping.jsonl'
G1=Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim')
OUT=ROOT/'humanml_g1_id_overlap_20261030'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    mapping={r['motion_id']:r for r in map(json.loads,MAP.open(encoding='utf8'))}
    splits={s:set((G1/(s+'.txt')).read_text().splitlines()) for s in ('train','val','test')}
    ids=set.union(*splits.values())
    overlap=ids & set(mapping)
    status=collections.Counter(mapping[i]['status'] for i in overlap)
    official_split=collections.Counter(mapping[i]['official_split'] for i in overlap)
    rows=[{'id':i,'g1_split':s,'official_split':mapping[i]['official_split'] if i in mapping else None,
           'raw_mapping_status':mapping[i]['status'] if i in mapping else None}
          for s,subset in splits.items() for i in sorted(subset)]
    OUT.mkdir(mode=0o700)
    with (OUT/'ids.jsonl').open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    summary={'g1_total_ids':len(ids),'g1_split_sizes':{s:len(v) for s,v in splits.items()},
             'g1_overlap_with_humanml_ids':len(overlap),'g1_not_in_humanml':len(ids-set(mapping)),
             'raw_mapping_status_for_overlap':dict(status),'official_splits_for_overlap':dict(official_split),
             'g1_id_official_split_mismatch':sum(r['official_split'] is not None and r['official_split']!=r['g1_split'] for r in rows),
             'mapping_sha256':digest(MAP),'g1_split_sha256':{s:digest(G1/(s+'.txt')) for s in splits},
             'limitations':['ID equality alone does not establish same raw recording, time alignment, physical quality, or v2 representational equivalence.',
                            'Existing G1 splits are project-specific; source-group disjointness not established.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
