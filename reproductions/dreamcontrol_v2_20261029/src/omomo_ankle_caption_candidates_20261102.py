"""Generate narrow, explicitly machine-labeled ankle-motion candidates.

These are descriptions of FK joint positions, not original OMOMO captions,
contact labels, or training approval. Thresholds are project choices, not
DreamControl-v2 author criteria.
"""
import argparse
from collections import Counter
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MANIFEST = ROOT / 'omomo_mixed_encoding_preview_20261101/manifest.json'
MANIFEST_SHA = 'd291aa0985304deaa9007df31c8700985109e2be6feac950c3ebd3766e808e7a'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def choose(points):
    # HumanML joint order: 7 left ankle, 8 right ankle. Use a 5-sample mean
    # to suppress one-frame outliers; report its effective frame centers.
    curves = np.stack([np.convolve(points[:, j, 2], np.ones(5)/5, mode='valid')
                       for j in (7, 8)], axis=0)
    options = []
    for side, c in zip(('left', 'right'), curves):
        k = int(np.argmax(c)); rise = float(c[k]-c[0]); fall = float(c[k]-c[-1])
        if k >= 4 and k <= len(c)-5 and rise >= .12 and fall >= .12:
            options.append((min(rise,fall), side, k, rise, fall, float(c[0]), float(c[k]), float(c[-1])))
    if not options: return None
    return max(options)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    if sha(MANIFEST)!=MANIFEST_SHA: raise ValueError('manifest hash mismatch')
    records=json.loads(MANIFEST.read_text())['records']
    if len(records)!=1717 or len({r['sample_id'] for r in records})!=1717: raise ValueError('manifest count/IDs')
    if shutil.disk_usage(ROOT).free < 8*1024**3+64*1024**2: raise RuntimeError('disk budget')
    a.out.mkdir(exist_ok=False); counts=Counter(); failures=[]; candidates=0
    with (a.out/'records.jsonl').open('x',encoding='utf8',buffering=1) as out:
      for i,row in enumerate(records):
        try:
          state=(ROOT/row['state_path']).resolve(strict=True); feature=(ROOT/row['feature_path']).resolve(strict=True)
          if not state.is_relative_to(ROOT) or not feature.is_relative_to(ROOT):raise ValueError('outside root')
          if sha(state)!=row['state_sha256'] or sha(feature)!=row['feature_sha256']:raise ValueError('hash mismatch')
          with np.load(state,allow_pickle=False) as z:
            points=z['points_z_up'].copy(); times=z['sample_times'].copy()
          with np.load(feature,allow_pickle=False) as z: nfeatures=len(z['features'])
          # Feature f describes consecutive states f and f+1. Do not claim
          # that the source's actual wall clock has been verified.
          if points.shape!=(nfeatures+1,22,3) or len(times)!=len(points) or len(points)<15 or not np.isfinite(points).all():raise ValueError('invalid state')
          if not np.allclose(times,np.arange(len(times))/20,atol=1e-10,rtol=0):raise ValueError('time grid')
          chosen=choose(points)
          if chosen is not None:
            strength,side,k,rise,fall,start,peak,end=chosen; candidates+=1
            label=f'The humanoid raises and lowers its {side} ankle.'
            result={'sample_id':row['sample_id'],'source_member':row['source_member'],'source_sha256':row['source_sha256'],
              'state_path':row['state_path'],'state_sha256':row['state_sha256'],'feature_path':row['feature_path'],
              'feature_sha256':row['feature_sha256'],'nominal_subject':row['nominal_subject'],
              'nominal_action_key':row['nominal_action_key'],'auxiliary_split':row['auxiliary_split'],
              'original_caption':row['caption'],'original_caption_accepted':False,'candidate_caption':label,
              'caption_origin':'machine_geometry_v1','state_frame_count':len(points),'feature_frame_count':nfeatures,
              'evidence':{'joint_index':7 if side=='left' else 8,'window_frames':5,'center_indices':[2,k+2,len(points)-3],
                          'nominal_seconds':[.05*(2),.05*(k+2),.05*(len(points)-3)],
                          'mean_z_m':[start,peak,end],'rise_m':rise,'fall_m':fall,'threshold_m':.12},
              'caption_accepted':False,'training_ready':False}
            counts[row['auxiliary_split']]+=1; out.write(json.dumps(result,allow_nan=False)+'\n')
        except Exception as e:
          failures.append({'sample_id':row['sample_id'],'error':repr(e)})
        if (i+1)%100==0 and shutil.disk_usage(ROOT).free < 8*1024**3+64*1024**2:raise RuntimeError('disk budget')
    if sha(MANIFEST)!=MANIFEST_SHA:raise RuntimeError('manifest changed')
    result={'manifest_sha256':MANIFEST_SHA,'script_sha256':sha(Path(__file__)),'scanned':len(records),'errors':failures,
      'candidate_count':candidates,'counts_by_split':dict(counts),'records_sha256':sha(a.out/'records.jsonl'),
      'caption_origin':'machine_geometry_v1','caption_accepted':False,'training_ready':False,
      'limitations':['5-frame averaged ankle FK height change, not sole position, foot contact or a physics check.',
        'Automatically generated project captions replace neither original OMOMO labels nor independent semantic review.',
        'Thresholds are explicit project choices, not author v2 filtering.',
        'Nominal sampling times and grouping do not establish real recording clocks or independent identities.']}
    with (a.out/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
