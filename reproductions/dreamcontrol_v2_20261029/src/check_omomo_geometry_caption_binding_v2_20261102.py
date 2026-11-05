"""Read-only smoke of one machine-geometry caption per source sample."""
import hashlib,json
from collections import Counter
from pathlib import Path
from audit_geometry_caption_binding import audit_geometry_captions
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'omomo_additional_geometry_captions_20261102'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    path=BASE/'records.jsonl';digest=sha(path);chosen={}
    for line in path.read_text().splitlines():
        entry=json.loads(line);chosen.setdefault(entry['sample_id'],entry)
    rows=[{'sample_id':x['sample_id'],'caption_origin':'machine_geometry_v2','caption':x['candidate_caption'],
           'feature_sha256':x['feature_sha256'],'caption_accepted':True} for x in chosen.values()]
    result=audit_geometry_captions(rows,evidence_manifest=path,evidence_sha256=digest,state_root=ROOT)
    result.pop('cases');result.update({'temporary_in_memory_only':True,'unique_sample_count':len(rows),
        'kinds':dict(Counter(x['kind'] for x in chosen.values())),'script_sha256':sha(Path(__file__)),'training_ready':False})
    with (BASE/'binding_smoke.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
