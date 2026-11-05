"""Independently sum train-only stats for the unaccepted geometry preview."""
from collections import Counter
import hashlib,json
from pathlib import Path
import numpy as np,torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'omomo_geometry_schema_preview_20261102'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(1);r=json.loads((BASE/'report.json').read_text());mh=sha(BASE/'manifest.json')
    for key,name in (('manifest_sha256','manifest.json'),('state_manifest_sha256','state_manifest.json'),
                     ('stats_sha256','stats.npz'),('geometry_receipts_sha256','geometry_receipts.jsonl')):
        if sha(BASE/name)!=r[key]:raise ValueError('preview output changed')
    rows=json.loads((BASE/'manifest.json').read_text())['records'];states=json.loads((BASE/'state_manifest.json').read_text())['records']
    receipts=[json.loads(x) for x in (BASE/'geometry_receipts.jsonl').read_text().splitlines()]
    si={x['sample_id']:x for x in states};ri={x['sample_id']:x for x in receipts}
    if len(si)!=len(rows) or len(ri)!=len(rows) or len({x['sample_id'] for x in rows})!=len(rows):raise ValueError('duplicate')
    sf=np.zeros(263);qf=np.zeros(263);sp=np.zeros(66);qp=np.zeros(66);n=0;counts=Counter();frames=Counter();groups={};digests={}
    for row in rows:
        sid=row['sample_id'];state=si[sid];receipt=ri[sid];f=ROOT/row['feature_path'];s=ROOT/state['state_path']
        if (row['caption_accepted'] is not False or row['training_ready'] is not False or
            receipt['candidate_caption']!=row['caption'] or receipt['caption_accepted'] is not False or
            receipt['state_sha256']!=state['state_sha256'] or receipt['feature_sha256']!=row['feature_sha256'] or
            sha(f)!=row['feature_sha256'] or sha(s)!=state['state_sha256']):raise ValueError('candidate binding')
        if groups.setdefault(row['source_id'],row['split'])!=row['split']:raise ValueError('nominal source crosses splits')
        with np.load(f,allow_pickle=False) as z:x=np.asarray(z['features'],dtype=np.float32)
        digest=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        if digests.setdefault(digest,row['split'])!=row['split']:raise ValueError('feature duplicate across splits')
        counts[row['split']]+=1;frames[row['split']]+=len(x)
        if row['split']=='train':
            a=x.astype(np.float64)
            with torch.no_grad():p=recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64)
            sf+=a.sum(0);qf+=(a*a).sum(0);sp+=p.sum(0);qp+=(p*p).sum(0);n+=len(x)
    means=(sf/n,sp/n);stds=(np.maximum(np.sqrt(np.maximum(qf/n-means[0]**2,0)),.001),
                            np.maximum(np.sqrt(np.maximum(qp/n-means[1]**2,0)),.001))
    with np.load(BASE/'stats.npz',allow_pickle=False) as stats:
        errors={k:float(np.max(np.abs(stats[k]-v))) for k,v in
                (('feature_mean',means[0]),('feature_std',stds[0]),('spatial_mean',means[1]),('spatial_std',stds[1]))}
    if max(errors.values())>1e-5 or dict(counts)!=r['counts_by_split'] or dict(frames)!=r['frames_by_split'] or sha(BASE/'manifest.json')!=mh:
        raise ValueError('statistics or count mismatch')
    result={'manifest_sha256':mh,'preview_report_sha256':sha(BASE/'report.json'),'script_sha256':sha(Path(__file__)),
            'counts_by_split':dict(counts),'frames_by_split':dict(frames),'statistics_max_abs_difference':errors,
            'duplicate_features_across_splits':0,'caption_accepted':0,'training_ready':False,
            'limitations':['Statistics use an independent sum but the same HumanML decoder.',
               'Nominal IDs and exact bytes cannot establish independent people or recordings.',
               'This is not evidence of original text, license, clock, contact or dynamics.']}
    with (BASE/'audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
