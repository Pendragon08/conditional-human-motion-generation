"""Independently recompute all wrist and pelvis selection predicates."""
from collections import Counter
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'omomo_additional_geometry_captions_20261102'
MANIFEST=ROOT/'omomo_mixed_encoding_preview_20261101/manifest.json'
SPECS={'left_wrist':(20,1,'The humanoid raises and lowers its left wrist.'),
       'right_wrist':(21,1,'The humanoid raises and lowers its right wrist.'),
       'pelvis':(0,-1,'The humanoid lowers and raises its pelvis.')}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    r=json.loads((BASE/'report.json').read_text());mh=sha(MANIFEST);rp=BASE/'records.jsonl'
    if mh!=r['manifest_sha256'] or sha(rp)!=r['records_sha256']:raise ValueError('hash mismatch')
    rows=[json.loads(x) for x in rp.read_text().splitlines()];idx={(x['sample_id'],x['kind']):x for x in rows}
    if len(idx)!=len(rows):raise ValueError('duplicate')
    sources=json.loads(MANIFEST.read_text())['records'];n=0;max_diff=0.;originals=0;samples=set();groups=set();counts=Counter()
    for src in sources:
      state=ROOT/src['state_path'];feature=ROOT/src['feature_path']
      if sha(state)!=src['state_sha256'] or sha(feature)!=src['feature_sha256']:raise ValueError('artifact hash')
      with np.load(state,allow_pickle=False) as f:p=f['points_z_up'].copy();t=f['sample_times'].copy()
      for kind,(joint,direction,label) in SPECS.items():
        vals=[direction*sum(float(p[i+j,joint,2]) for j in range(5))/5 for i in range(len(p)-4)]
        peak=max(vals);k=vals.index(peak);up=peak-vals[0];down=peak-vals[-1]
        eligible=k>=4 and k<=len(vals)-5 and up>=.12 and down>=.12
        entry=idx.get((src['sample_id'],kind))
        if eligible!=(entry is not None):raise ValueError('selection mismatch')
        if not eligible:continue
        e=entry['evidence'];centers=[2,k+2,len(p)-3]
        if (entry['candidate_caption']!=label or entry['original_caption']!=src['caption'] or
            entry['state_sha256']!=src['state_sha256'] or entry['feature_sha256']!=src['feature_sha256'] or
            e['center_indices']!=centers or e['joint_index']!=joint or e['direction']!=direction or
            not np.allclose(e['nominal_seconds'],t[centers],atol=1e-12,rtol=0)):raise ValueError('binding mismatch')
        diff=max(abs(a-b) for a,b in zip(e['oriented_mean_z_m']+[e['rise_m'],e['fall_m']],[vals[0],peak,vals[-1],up,down]))
        if diff>1e-7:raise ValueError('numeric mismatch')
        max_diff=max(max_diff,diff);n+=1;counts[kind]+=1;counts['split_'+src['auxiliary_split']]+=1
        samples.add(src['sample_id']);groups.add(src['nominal_action_key']);originals+=src['source_member'].endswith('_original.npz')
    if n!=len(rows) or n!=r['labels']:raise ValueError('count mismatch')
    result={'source_manifest_sha256':mh,'records_sha256':sha(rp),'audit_script_sha256':sha(Path(__file__)),
            'checked_samples':len(sources),'labels':n,'unique_samples':len(samples),'nominal_action_groups':len(groups),
            'original_labels':originals,'variant_labels':n-originals,'counts':dict(counts),'max_scalar_difference_m':max_diff,
            'caption_accepted':False,'training_ready':False,
            'limitations':['Checks numerical predicate and hashes, not human semantic intent, contact, licensing or real clock.',
                           'Counts are labels, not independent subjects or training-ready samples.']}
    with (BASE/'audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
