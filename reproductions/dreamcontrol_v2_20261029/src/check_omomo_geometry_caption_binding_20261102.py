"""Read-only batch smoke for geometry caption bindings; does not accept data."""
import hashlib
import json
from pathlib import Path
from audit_geometry_caption_binding import audit_geometry_captions

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'omomo_ankle_caption_candidates_20261102'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    path=BASE/'records.jsonl';receipt=sha(path)
    candidates=[json.loads(line) for line in path.read_text().splitlines()]
    rows=[{'sample_id':x['sample_id'],'caption_origin':'machine_geometry_v1',
           'caption':x['candidate_caption'],'feature_sha256':x['feature_sha256'],
           'caption_accepted':True} for x in candidates]
    result=audit_geometry_captions(rows,evidence_manifest=path,evidence_sha256=receipt,state_root=ROOT)
    result.update({'example_acceptance_declarations':'in_memory_smoke_only',
                   'candidate_count':len(rows),'script_sha256':sha(Path(__file__)),
                   'training_ready':False})
    result.pop('cases')
    with (BASE/'binding_smoke.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
