"""Read back uncropped review references and compare them to prior 10 s previews."""
import hashlib,json,zipfile,io
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
O=R/'omni29_uncropped_review_20261103';D=R/'grab_omni_mixed29_full_preview_20261103';Z=R/'data/OmniRetarget_Dataset/robot-object.zip'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 report=json.loads((O/'report.json').read_text());records=O/'records.jsonl';m=D/'manifest.json'
 if sha(records)!=report['records_sha256'] or sha(m)!=report['manifest_sha256']:raise ValueError('binding')
 rows=[json.loads(x) for x in records.read_text().splitlines()];orig={r['sample_id']:r for r in json.loads(m.read_text())['records']}
 if len(rows)!=80 or len({x['sample_id'] for x in rows})!=80:raise ValueError('coverage')
 details=[];maxstate=maxfeature=0.
 with zipfile.ZipFile(Z) as z:
  for row in rows:
   c=orig[row['sample_id']];full=O/row['output'];short=R/c['state_path'];raw=z.read(row['source_member'])
   if sha(full)!=row['output_sha256'] or sha(short)!=row['prior_preview_sha256'] or hashlib.sha256(raw).hexdigest()!=row['source_sha256'] or c['caption']!=row['caption']:raise ValueError('hash or text')
   with np.load(io.BytesIO(raw),allow_pickle=False) as f:n=len(f['qpos']);fps=float(f['fps'])
   with np.load(full,allow_pickle=False) as f:
    q=np.asarray(f['qpos29_source_layout_20hz']);x=np.asarray(f['features']);p=np.asarray(f['points_z_up']);times=np.asarray(f['sample_times']);names=f['joint_names']
   with np.load(short,allow_pickle=False) as f:s=np.asarray(f['qpos29_source_layout_20hz']);sx=np.asarray(f['features'])
   if fps!=30 or q.shape!=(len(times),43) or p.shape!=(len(times),22,3) or x.shape!=(len(times)-1,263) or len(names)!=29 or len(s)!=200:raise ValueError('shape')
   if not all(np.isfinite(v).all() for v in (q,x,p,times)) or not np.allclose(times,np.arange(len(times))/20.,atol=1e-12,rtol=0):raise ValueError('time or nonfinite')
   if times[-1]>(n-1)/fps+1e-9 or (n-1)/fps-times[-1]>=.05+1e-9:raise ValueError('coverage')
   ds=float(np.max(np.abs(q[:len(s)]-s)));dx=float(np.max(np.abs(x[:len(sx)]-sx)))
   if ds>1e-10:raise ValueError('state prefix mismatch')
   maxstate=max(maxstate,ds);maxfeature=max(maxfeature,dx)
   details.append({'id':row['sample_id'],'states':len(q),'seconds':float(times[-1]),'state_prefix_max_abs':ds,'feature_prefix_max_abs':dx,'artifact_sha256':row['output_sha256']})
 out={'records_sha256':sha(records),'clips':len(details),'state_prefix_max_abs':maxstate,'feature_prefix_max_abs':maxfeature,
      'full_duration_s_min':min(x['seconds'] for x in details),'full_duration_s_max':max(x['seconds'] for x in details),
      'caption_accepted':0,'training_ready':False,'details':details,
      'limitations':['Readback/time and prefix checks are technical, not caption or source-license acceptance.',
                     'Full references may exceed paper model input length and are for matching/review, not direct training.']}
 with (O/'audit.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
 print(json.dumps({k:v for k,v in out.items() if k!='details'}))
if __name__=='__main__':main()
