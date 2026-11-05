"""Recheck existence of paths cited in TextOP's previously generated inventories."""
import collections
import hashlib
import json
from pathlib import Path

TEXTOP=Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/data_preparation_20261025')
OUT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/textop_pair_path_audit_20261030')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    files=['available_raw_pairs.private.json','additional_existing_raw_pairs.private.json']
    result=[]
    for fn in files:
        entries=json.loads((TEXTOP/fn).read_text())
        for e in entries:
            paths=e['raw_stageii_candidates']
            existing=[p for p in paths if Path(p).is_file()]
            result.append({'inventory':fn,'split':e['split'],'source':e['source'],
                           'candidate_count':len(paths),'existing_candidate_count':len(existing)})
    OUT.mkdir(mode=0o700)
    with (OUT/'rows.jsonl').open('x',encoding='utf8') as f:
        for r in result:f.write(json.dumps(r)+'\n')
    summary={'records':len(result),'with_existing_candidate':sum(x['existing_candidate_count']>0 for x in result),
             'without_existing_candidate':sum(x['existing_candidate_count']==0 for x in result),
             'by_inventory':dict(collections.Counter(x['inventory'] for x in result)),
             'inputs_sha256':{p:sha(TEXTOP/p) for p in files},'rows_sha256':sha(OUT/'rows.jsonl'),
             'limitations':['Only prior candidate manifests rechecked, not all 8291 sequences or other source directories.',
                            'File existence does not establish identity, hash, timeline, physical filtering or 29-DoF equivalence.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
