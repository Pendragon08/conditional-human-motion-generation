"""Build a non-training-ready 29DoF/263D Omni text schema and train-only stats.

Nominal text/group associations are not accepted labels or official splits.
No physical-action filtering is applied; sources and prior previews are read-only.
"""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,torch
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path.insert(0,str(T))
from data_loaders.humanml.scripts.motion_process import recover_from_ric
class Moments:
 def __init__(self,n):self.n=0;self.mean=np.zeros(n);self.m2=np.zeros(n)
 def add(self,x):
  x=np.asarray(x,dtype=np.float64);n=len(x);mu=x.mean(axis=0);m2=((x-mu)**2).sum(axis=0);d=mu-self.mean;total=self.n+n
  self.mean+=d*n/total;self.m2+=m2+d*d*self.n*n/total;self.n=total
 def result(self):return self.mean,np.maximum(np.sqrt(np.maximum(self.m2/self.n,0)),.001)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);out=R/'omni29_text_training_preview_20261103';coverage=R/'omni29_text_motion_coverage_audit_20261103.json';c=json.loads(coverage.read_text())
 if out.exists():raise FileExistsError(out)
 if c['total_candidates']!=1826 or c['accepted_captions']!=0:raise ValueError('coverage not expected')
 group=R/'omomo_group_split_20261101/candidates.jsonl'
 if sha(group)!=c['candidate_manifest_sha256']:raise ValueError('group changed')
 candidates=[json.loads(x) for x in group.read_text(encoding='utf8').splitlines()];index={}
 for folder in ('omni29_text_motion_pilot10_20261103','omni29_text_motion_batch_a_20261103','omni29_text_motion_batch_b_20261103'):
  d=R/folder;e=c['evidence'][folder]
  if sha(d/'records.jsonl')!=e['records_sha256'] or sha(d/'audit.json')!=e['audit_sha256']:raise ValueError('batch evidence changed')
  for line in (d/'records.jsonl').read_text(encoding='utf8').splitlines():
   row=json.loads(line);i=row['index']
   if i in index or row['status']!='technical_preview':raise ValueError('batch coverage')
   index[i]={'path':f'{folder}/{row["artifact"]}','hash':row['artifact_sha256'],'source_sha256':row['source_sha256'],'annotation_sha256':row['annotation_sha256'],'caption':row['caption'],'cropped':row['cropped_for_max_10s']}
 pilot=R/'omni29_direct_text_motion_pilot_20261103';pr=json.loads((pilot/'report.json').read_text());e=c['evidence'][pilot.name]
 if sha(pilot/'audit_v2.json')!=e['audit_sha256'] or sha(pilot/'state_features.npz')!=e['artifact_sha256']:raise ValueError('pilot evidence changed')
 index[203]={'path':f'{pilot.name}/state_features.npz','hash':pr['output_sha256'],'source_sha256':pr['source_member_sha256'],'annotation_sha256':pr['annotation_sha256'],'caption':pr['caption'],'cropped':False}
 if set(index)!=set(range(1826)):raise ValueError('coverage mismatch')
 fm,sm=Moments(263),Moments(66);records=[]
 for i,cand in enumerate(candidates):
  if i%32==0 and shutil.disk_usage(R).free<8*1024**3+256*1024**2:raise RuntimeError('disk reserve')
  row=index[i];p=R/row['path']
  if sha(p)!=row['hash'] or row['source_sha256']!=cand['motion_member_sha256'] or row['annotation_sha256']!=cand['annotation_sha256'] or row['caption']!=cand['caption']:raise ValueError('binding mismatch')
  with np.load(p,allow_pickle=False) as f:x=np.asarray(f['features'],dtype=np.float32)
  if x.ndim!=2 or x.shape[1]!=263 or not 2<=len(x)<=199 or not np.isfinite(x).all():raise ValueError('features')
  groupid='OMOMO_nominal_subject:'+cand['nominal_subject'];sourceid='OMOMO_nominal_action:'+cand['nominal_action_key']
  records.append({'sample_id':f'omni29_{i:04d}','group_id':groupid,'source_id':sourceid,'split':cand['auxiliary_split'],'provenance_keys':{'nominal_subject':[groupid],'nominal_action':[sourceid],'source_member_sha256':[row['source_sha256']]},'caption':row['caption'],'caption_source':'OMOMO author archive: '+cand['annotation_member'],'caption_origin':'source_text','caption_accepted':False,'annotation_sha256':row['annotation_sha256'],'feature_path':row['path'],'feature_sha256':row['hash'],'state_path':row['path'],'state_sha256':row['hash'],'fps':20.,'source_member':cand['motion_member'],'source_sha256':row['source_sha256'],'encoding_variant':'sign_continuous_v1','cropped_for_max_10s':row['cropped'],'training_ready':False})
  if cand['auxiliary_split']=='train':
   with torch.no_grad():pts=recover_from_ric(torch.from_numpy(x),22).numpy()
   if pts.shape!=(len(x),22,3) or not np.isfinite(pts).all():raise ValueError('decode failure')
   fm.add(x);sm.add(pts.reshape(len(x),66))
  if (i+1)%200==0:print('samples',i+1,flush=True)
 if fm.n!=124594 or sm.n!=fm.n:raise ValueError('training frame count')
 out.mkdir(exist_ok=False);fmean,fstd=fm.result();smean,sstd=sm.result()
 with (out/'manifest.json').open('x',encoding='utf8') as f:json.dump({'records':records,'training_ready':False},f,ensure_ascii=False,allow_nan=False)
 with (out/'stats.npz').open('xb') as f:np.savez(f,feature_mean=fmean,feature_std=fstd,spatial_mean=smean,spatial_std=sstd)
 result={'coverage_sha256':sha(coverage),'manifest_sha256':sha(out/'manifest.json'),'stats_sha256':sha(out/'stats.npz'),'train_frames':fm.n,'accepted_captions':0,'training_ready':False,'limitations':['Nominal source captions are not accepted and grouping is not independently established.','Stats use provisional data and must be recomputed after final acceptance.','No physical-action filtering applied to Omni; not a training run.']}
 with (out/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
