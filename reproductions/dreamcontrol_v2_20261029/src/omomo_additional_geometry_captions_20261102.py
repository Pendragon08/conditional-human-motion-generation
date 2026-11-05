"""Supplemental wrist and pelvis height captions; not original source text."""
from collections import Counter
import hashlib,json,shutil
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
MANIFEST=ROOT/'omomo_mixed_encoding_preview_20261101/manifest.json'
EXPECTED='d291aa0985304deaa9007df31c8700985109e2be6feac950c3ebd3766e808e7a'
OUT=ROOT/'omomo_additional_geometry_captions_20261102'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()
def detect(points,joint,direction):
    c=np.convolve(points[:,joint,2],np.ones(5)/5,mode='valid')*direction
    k=int(np.argmax(c));rise=float(c[k]-c[0]);fall=float(c[k]-c[-1])
    if k<4 or k>len(c)-5 or rise<.12 or fall<.12:return None
    return {'joint_index':joint,'direction':direction,'window_frames':5,'threshold_m':.12,
            'center_indices':[2,k+2,len(points)-3],'nominal_seconds':[.1,(k+2)/20,(len(points)-3)/20],
            'oriented_mean_z_m':[float(c[0]),float(c[k]),float(c[-1])],'rise_m':rise,'fall_m':fall}
def main():
    if sha(MANIFEST)!=EXPECTED or shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:raise RuntimeError('input or budget')
    records=json.loads(MANIFEST.read_text())['records']
    if len(records)!=1717:raise ValueError('manifest count')
    OUT.mkdir(exist_ok=False);counts=Counter();groups=set();samples=set();errors=[]
    with (OUT/'records.jsonl').open('x',encoding='utf8',buffering=1) as f:
      for i,r in enumerate(records):
        try:
          sp=(ROOT/r['state_path']).resolve(strict=True);fp=(ROOT/r['feature_path']).resolve(strict=True)
          if not sp.is_relative_to(ROOT) or not fp.is_relative_to(ROOT) or sha(sp)!=r['state_sha256'] or sha(fp)!=r['feature_sha256']:raise ValueError('hash or path')
          with np.load(sp,allow_pickle=False) as z:p=z['points_z_up'].copy();t=z['sample_times'].copy()
          with np.load(fp,allow_pickle=False) as z:n=len(z['features'])
          if p.shape!=(n+1,22,3) or len(p)<15 or not np.isfinite(p).all() or not np.allclose(t,np.arange(len(p))/20,rtol=0,atol=1e-10):raise ValueError('invalid points/time')
          for name,joint,direction,label in (('left_wrist',20,1,'The humanoid raises and lowers its left wrist.'),
                    ('right_wrist',21,1,'The humanoid raises and lowers its right wrist.'),
                    ('pelvis',0,-1,'The humanoid lowers and raises its pelvis.')):
            evidence=detect(p,joint,direction)
            if evidence is None:continue
            data={'sample_id':r['sample_id'],'kind':name,'candidate_caption':label,'caption_origin':'machine_geometry_v2',
                  'source_member':r['source_member'],'source_sha256':r['source_sha256'],'state_path':r['state_path'],
                  'state_sha256':r['state_sha256'],'feature_path':r['feature_path'],'feature_sha256':r['feature_sha256'],
                  'nominal_subject':r['nominal_subject'],'nominal_action_key':r['nominal_action_key'],
                  'auxiliary_split':r['auxiliary_split'],'original_caption':r['caption'],'original_caption_accepted':False,
                  'evidence':evidence,'caption_accepted':False,'training_ready':False}
            f.write(json.dumps(data,allow_nan=False)+'\n');counts[name]+=1;counts['split_'+r['auxiliary_split']]+=1
            samples.add(r['sample_id']);groups.add(r['nominal_action_key'])
        except Exception as e:errors.append({'sample_id':r['sample_id'],'error':repr(e)})
        if (i+1)%100==0 and shutil.disk_usage(ROOT).free<8*1024**3+64*1024**2:raise RuntimeError('disk budget')
    if sha(MANIFEST)!=EXPECTED:raise RuntimeError('manifest changed')
    result={'manifest_sha256':EXPECTED,'script_sha256':sha(Path(__file__)),'scanned':len(records),'errors':errors,
            'labels':sum(counts[k] for k in ('left_wrist','right_wrist','pelvis')),'unique_samples':len(samples),
            'nominal_action_groups':len(groups),'counts':dict(counts),'records_sha256':sha(OUT/'records.jsonl'),
            'caption_accepted':False,'training_ready':False,
            'limitations':['Height of FK joint, averaged over five frames, not ground or contact.',
                'Machine descriptions are not source captions or visual review.',
                'Threshold is a project choice; recordings and split independence unverified.']}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
