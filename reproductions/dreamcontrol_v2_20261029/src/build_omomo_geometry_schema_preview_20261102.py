"""Build an unaccepted, hash-bound loader preview for narrow geometry labels.

Exactly one candidate per trajectory, never substitutes for OMOMO captions,
and no data or candidate is accepted for training by this script.
"""
import hashlib,json,shutil
from collections import Counter
from pathlib import Path
import numpy as np,torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from audit_robot_features import Moments,audit
from audit_geometry_caption_binding import audit_geometry_captions

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SCHEMA=ROOT/'omomo_training_schema_preview_20261101'
V1=ROOT/'omomo_ankle_caption_candidates_20261102'
V2=ROOT/'omomo_additional_geometry_captions_20261102'
OUT=ROOT/'omomo_geometry_schema_preview_20261102'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(1)
    if shutil.disk_usage(ROOT).free<8*1024**3+128*1024**2:raise RuntimeError('disk budget')
    schema_report=json.loads((SCHEMA/'report.json').read_text())
    if sha(SCHEMA/'manifest.json')!=schema_report['manifest_sha256'] or sha(SCHEMA/'state_manifest.json')!=schema_report['state_manifest_sha256']:
        raise ValueError('schema hash mismatch')
    old={x['sample_id']:x for x in json.loads((SCHEMA/'manifest.json').read_text())['records']}
    states={x['sample_id']:x for x in json.loads((SCHEMA/'state_manifest.json').read_text())['records']}
    if len(old)!=1717 or len(states)!=1717:raise ValueError('schema count')
    candidates={};pins={}
    # Prefer v2 wrist/pelvis descriptions; use v1 ankle only if no v2 label.
    # Dict insertion order retains the source's deterministic predicate order.
    for directory in (V2,V1):
        report=json.loads((directory/'report.json').read_text()); audit_report=json.loads((directory/'audit.json').read_text())
        source=directory/'records.jsonl';h=sha(source)
        if h!=report['records_sha256'] or h!=audit_report.get('records_sha256',audit_report.get('candidates_sha256')):
            raise ValueError('candidate audit mismatch')
        pins[str(directory.relative_to(ROOT))]=h
        for line in source.read_text().splitlines():
            row=json.loads(line);candidates.setdefault(row['sample_id'],row)
    if not candidates or not set(candidates).issubset(old):raise ValueError('candidate IDs')
    OUT.mkdir(exist_ok=False);records=[];state_rows=[];receipts=[];counts=Counter();frames=Counter();feature=Moments(263);spatial=Moments(66)
    for i,(sid,candidate) in enumerate(sorted(candidates.items())):
        r=dict(old[sid]);state=states[sid]
        if candidate['state_sha256']!=state['state_sha256'] or candidate['feature_sha256']!=r['feature_sha256']:
            raise ValueError('candidate state or feature changed')
        feature_path=(ROOT/r['feature_path']).resolve(strict=True);state_path=(ROOT/state['state_path']).resolve(strict=True)
        if not feature_path.is_relative_to(ROOT) or not state_path.is_relative_to(ROOT) or sha(feature_path)!=r['feature_sha256'] or sha(state_path)!=state['state_sha256']:
            raise ValueError('artifact hash mismatch')
        original=r['caption'];r.update({'caption':candidate['candidate_caption'],'caption_origin':candidate['caption_origin'],
          'caption_source':'generated from G1 FK, project threshold, not source annotation','caption_accepted':False,
          'original_caption':original,'original_caption_accepted':False,'training_ready':False})
        if candidate['original_caption']!=original:raise ValueError('original caption mismatch')
        with np.load(feature_path,allow_pickle=False) as z:x=np.asarray(z['features'],dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=263 or not np.isfinite(x).all():raise ValueError('bad features')
        if r['split']=='train':
            with torch.no_grad():p=recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66)
            feature.add(x);spatial.add(p)
        counts[r['split']]+=1;frames[r['split']]+=len(x);records.append(r);state_rows.append(state);receipts.append(candidate)
        if i%100==0 and shutil.disk_usage(ROOT).free<8*1024**3+128*1024**2:raise RuntimeError('disk budget')
    if set(counts)!={'train','val','test'}:raise ValueError('missing split')
    fm,fs=feature.values(.001);pm,ps=spatial.values(.001)
    stats={'feature_mean':fm,'feature_std':fs,'spatial_mean':pm,'spatial_std':ps}
    with (OUT/'manifest.json').open('x',encoding='utf8') as f:json.dump({'records':records,'training_ready':False},f,indent=2,allow_nan=False)
    with (OUT/'state_manifest.json').open('x',encoding='utf8') as f:json.dump({'records':state_rows,'training_ready':False},f,indent=2,allow_nan=False)
    with (OUT/'geometry_receipts.jsonl').open('x',encoding='utf8') as f:
        for row in receipts:f.write(json.dumps(row,allow_nan=False)+'\n')
    with (OUT/'stats.npz').open('xb') as f:np.savez_compressed(f,**stats)
    features=audit(ROOT,records,stats,min_std=.001)
    # Explicit acceptance is temporary and only exercises hash/timing binding.
    temp=[{**r,'caption_accepted':True} for r in records]
    binding=audit_geometry_captions(temp,evidence_manifest=OUT/'geometry_receipts.jsonl',
                                    evidence_sha256=sha(OUT/'geometry_receipts.jsonl'),state_root=ROOT)
    if not features['technical_checks_passed'] or not binding['geometry_checks_passed'] or features['training_ready'] or binding['training_ready']:
        raise ValueError('unexpected preflight outcome')
    result={'schema_manifest_sha256':schema_report['manifest_sha256'],'candidate_sha256':pins,
            'script_sha256':sha(Path(__file__)),'manifest_sha256':sha(OUT/'manifest.json'),
            'state_manifest_sha256':sha(OUT/'state_manifest.json'),'stats_sha256':sha(OUT/'stats.npz'),
            'geometry_receipts_sha256':sha(OUT/'geometry_receipts.jsonl'),'counts_by_split':dict(counts),'frames_by_split':dict(frames),
            'candidate_origins':dict(Counter(x['caption_origin'] for x in records)),
            'technical_checks_passed':True,'geometry_binding_checks_passed_in_memory':True,
            'captions_accepted':0,'training_ready':False,
            'limitations':['Auxiliary split and nominal provenance do not establish independent people or recordings.',
                           'Machine descriptions are not original OMOMO captions or full interaction semantics.',
                           'Time, applicable rights and quality acceptance remain unverified.',
                           'Spatial statistics and FK use the existing project decoder.']}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
