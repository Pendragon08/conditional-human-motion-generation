"""Compare 80 cropped previews with their source tails; descriptive, not semantic acceptance."""
import hashlib,io,json,math,shutil,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_mixed29_full_preview_20261103'
Z=R/'data/OmniRetarget_Dataset/robot-object.zip'
O=R/'omni29_cropped_tail_diagnostic_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if O.exists():raise FileExistsError(O)
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 manifest=D/'manifest.json';report=json.loads((D/'report.json').read_text())
 if sha(manifest)!=report['manifest_sha256']:raise ValueError('manifest hash')
 rows=[x for x in json.loads(manifest.read_text())['records'] if x['dataset_source']=='OmniRetarget/OMOMO' and x['cropped_for_max_10s']]
 if len(rows)!=80:raise ValueError('crop count')
 cases=[];groups=set()
 with zipfile.ZipFile(Z) as z:
  for r in rows:
   raw=z.read(r['source_member'])
   if hashlib.sha256(raw).hexdigest()!=r['source_sha256']:raise ValueError('source hash')
   with np.load(io.BytesIO(raw),allow_pickle=False) as f:
    q=np.asarray(f['qpos'],dtype=np.float64);fps=float(f['fps'])
   path=R/r['state_path']
   if sha(path)!=r['state_sha256']:raise ValueError('state hash')
   with np.load(path,allow_pickle=False) as f:out=np.asarray(f['qpos29_source_layout_20hz'],dtype=np.float64);t=np.asarray(f['sample_times'])
   if q.ndim!=2 or q.shape[1]!=43 or fps!=30 or len(out)!=200 or abs(t[-1]-9.95)>1e-9:raise ValueError('time/shape')
   start=math.ceil(float(t[-1])*fps-1e-9);tail=q[start:]
   if len(tail)<1:raise ValueError('missing source tail')
   obj=tail[:,40:43]-out[-1,40:43];root=tail[:,4:7]-out[-1,4:7]
   ob_norm=np.linalg.norm(obj,axis=1);root_norm=np.linalg.norm(root,axis=1)
   cases.append({'sample_id':r['sample_id'],'source_id':r['source_id'],'source_member':r['source_member'],
    'source_sha256':r['source_sha256'],'preview_sha256':r['state_sha256'],'caption':r['caption'],
    'preview_end_s':float(t[-1]),'source_last_time_s':(len(q)-1)/fps,
    'source_tail_frames':len(tail),'object_max_displacement_from_cut_m':float(ob_norm.max()),
    'object_end_displacement_from_cut_m':float(ob_norm[-1]),'root_max_displacement_from_cut_m':float(root_norm.max()),
    'object_cut_height_m':float(out[-1,42]),'object_source_end_height_m':float(q[-1,42])})
   groups.add(r['source_id'])
 values=np.array([x['object_max_displacement_from_cut_m'] for x in cases])
 out={'manifest_sha256':sha(manifest),'total':len(cases),'nominal_groups':len(groups),
      'object_tail_max_above_0_05m':int((values>.05).sum()),'object_tail_max_above_0_20m':int((values>.2).sum()),
      'object_tail_max_quantiles_m':{str(x):float(np.quantile(values,x)) for x in (0,.5,.9,1)},
      'limitations':['Object and root displacement are diagnostics, not semantic matching, contact, support or physical-action filters.',
       'No displacement threshold accepts or rejects a caption; motion after 9.95 s does not prove a specific action occurred there.',
       'Source clocks are nominal as published and frames include the first source time following the final preview time.'],
      'training_ready':False}
 O.mkdir(exist_ok=False)
 with (O/'records.jsonl').open('x',encoding='utf8') as f:
  for r in cases:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
 out['records_sha256']=sha(O/'records.jsonl')
 with (O/'report.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps(out))
if __name__=='__main__':main()
