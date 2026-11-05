"""Build full-length 20 Hz review references for the 80 previously truncated Omni clips.

These outputs are not 10-second model inputs and do not establish label truth.
All releases and the existing previews are read-only; no retargeting or physical
action screening is performed.
"""
import hashlib,io,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'grab_omni_mixed29_full_preview_20261103'
Z=R/'data/OmniRetarget_Dataset/robot-object.zip'
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
O=R/'omni29_uncropped_review_20261103'
sys.path[:0]=[str(R),str(R/'ssm_new6_20hz_resume_20261101'),str(R/'antialias_v3_20261030/src'),str(T)]
from g1_fk import URDFFK
from feature263_signfix import encode_signfixed
from resample_antialias_omniretarget import resample_antialias
NAMES=tuple(n+'_joint' for n in ('left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if O.exists():raise FileExistsError(O)
 if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
 report=json.loads((D/'report.json').read_text());m=D/'manifest.json'
 if sha(m)!=report['manifest_sha256'] or sha(U)!='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95':raise ValueError('input hash')
 rows=[x for x in json.loads(m.read_text())['records'] if x['dataset_source']=='OmniRetarget/OMOMO' and x['cropped_for_max_10s']]
 if len(rows)!=80:raise ValueError('unexpected crop count')
 fk=URDFFK(U);names=NAMES
 if len(fk.actuated)!=29 or set(fk.actuated)!=set(names):raise ValueError('joint names')
 O.mkdir(exist_ok=False);records=[]
 with zipfile.ZipFile(Z) as z:
  for i,r in enumerate(rows):
   if shutil.disk_usage(R).free<8*1024**3+356*1024**2:raise RuntimeError('disk reserve')
   raw=z.read(r['source_member'])
   if hashlib.sha256(raw).hexdigest()!=r['source_sha256']:raise ValueError('source hash')
   with np.load(io.BytesIO(raw),allow_pickle=False) as f:q=np.asarray(f['qpos']);fps=float(f['fps'])
   if q.ndim!=2 or q.shape[1]!=43 or fps!=30.:raise ValueError('source format')
   sampled=resample_antialias(q,fps,20.,max_seconds=None)
   points=fk.forward(sampled[:,4:7],sampled[:,[1,2,3,0]],sampled[:,7:36],names)
   feature=encode_signfixed(points,foot_velocity_squared_threshold=.002)
   if len(sampled)<=200 or feature.shape!=(len(sampled)-1,263) or not np.isfinite(feature).all():raise ValueError('unexpected length')
   p=O/('sample_%04d.npz'%i)
   with p.open('xb') as f:np.savez_compressed(f,qpos29_source_layout_20hz=sampled,points_z_up=points,features=feature,sample_times=np.arange(len(sampled))/20.,joint_names=np.asarray(names))
   records.append({'sample_id':r['sample_id'],'nominal_source_group':r['source_id'],'source_member':r['source_member'],
       'source_sha256':r['source_sha256'],'prior_preview_sha256':r['state_sha256'],'caption':r['caption'],
       'annotation_sha256':r['annotation_sha256'],'source_frames':len(q),'source_fps_nominal':fps,
       'sampled_states':len(sampled),'output':p.name,'output_sha256':sha(p),'caption_accepted':False})
  with (O/'records.jsonl').open('x',encoding='utf8') as f:
   for x in records:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
 out={'manifest_sha256':sha(m),'records_sha256':sha(O/'records.jsonl'),'clips':len(records),'nominal_source_groups':len(set(x['nominal_source_group'] for x in records)),
      'training_ready':False,'limitations':['Full-duration review references can exceed the 10-second model input and are not training-ready.',
      'Resampling/filtering is approximate and source clock is nominal.',
      'Source caption, event timing, license scope and semantics are unverified. No physical-action filtering.']}
 with (O/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps(out))
if __name__=='__main__':main()
