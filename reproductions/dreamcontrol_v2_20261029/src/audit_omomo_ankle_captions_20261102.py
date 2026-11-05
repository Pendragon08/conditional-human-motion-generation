"""Recompute narrow ankle caption candidates using scalar summation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'omomo_ankle_caption_candidates_20261102'
MANIFEST=ROOT/'omomo_mixed_encoding_preview_20261101/manifest.json'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def scalar(points):
    out=[]; n=len(points)
    for side,joint in (('left',7),('right',8)):
        vals=[sum(float(points[i+d,joint,2]) for d in range(5))/5 for i in range(n-4)]
        peak=max(vals); k=vals.index(peak); rise=peak-vals[0];fall=peak-vals[-1]
        if k>=4 and k<=len(vals)-5 and rise>=.12 and fall>=.12:
            out.append((min(rise,fall),side,k,rise,fall,vals[0],peak,vals[-1]))
    return max(out) if out else None

def main():
    report=json.loads((BASE/'report.json').read_text()); mh=sha(MANIFEST)
    if mh!=report['manifest_sha256'] or sha(BASE/'records.jsonl')!=report['records_sha256']:raise ValueError('input hash mismatch')
    source=json.loads(MANIFEST.read_text())['records']
    candidates=[json.loads(x) for x in (BASE/'records.jsonl').read_text().splitlines()]
    lookup={r['sample_id']:r for r in candidates}
    if len(lookup)!=len(candidates):raise ValueError('duplicate candidate')
    counts=Counter(); groups=set(); originals=0; maximum=0.; total=0
    for row in source:
        p=ROOT/row['state_path']; q=ROOT/row['feature_path']
        if sha(p)!=row['state_sha256'] or sha(q)!=row['feature_sha256']:raise ValueError('changed artifact')
        with np.load(p,allow_pickle=False) as f: points=f['points_z_up'].copy(); times=f['sample_times'].copy()
        calculated=scalar(points); found=lookup.get(row['sample_id'])
        if (calculated is None)!=(found is None):raise ValueError('candidate selection mismatch')
        if found is None:continue
        _,side,k,rise,fall,start,peak,end=calculated
        if found['candidate_caption']!=f'The humanoid raises and lowers its {side} ankle.':raise ValueError('caption mismatch')
        if found['original_caption']!=row['caption'] or found['state_sha256']!=row['state_sha256'] or found['feature_sha256']!=row['feature_sha256']:raise ValueError('binding mismatch')
        e=found['evidence']; centers=[2,k+2,len(points)-3]
        if e['center_indices']!=centers or e['joint_index']!=(7 if side=='left' else 8):raise ValueError('index mismatch')
        if not np.allclose(e['nominal_seconds'],times[centers],atol=1e-12,rtol=0):raise ValueError('time mismatch')
        delta=max(abs(a-b) for a,b in zip(e['mean_z_m']+[e['rise_m'],e['fall_m']],[start,peak,end,rise,fall]))
        if delta>1e-7:raise ValueError('numeric mismatch')
        maximum=max(maximum,delta);counts[row['auxiliary_split']]+=1;groups.add(row['nominal_action_key'])
        originals+=row['source_member'].endswith('_original.npz');total+=1
    if total!=len(candidates) or total!=report['candidate_count']:raise ValueError('count mismatch')
    result={'source_manifest_sha256':mh,'candidates_sha256':sha(BASE/'records.jsonl'),
            'audit_script_sha256':sha(Path(__file__)),'checked_all':len(source),'candidate_count':total,
            'original_count':originals,'variant_count':total-originals,'nominal_action_groups':len(groups),
            'counts_by_split':dict(counts),'max_scalar_difference_m':maximum,
            'caption_accepted':False,'training_ready':False,
            'limitations':['Scalar sums independently check selection, not ground truth or semantic quality.',
                           'HumanML ankle FK joint is not a foot or contact indicator.',
                           'No recording identity, licensing, physical, or training acceptance.']}
    with (BASE/'audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
