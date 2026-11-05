"""Read-only time-sensitivity comparison: pre-existing human mirrors vs G1.

The 20Hz human mirror is third-party; source identity and retargeting can differ.
Output is a diagnostic, not an inferred or corrected clock or training acceptance.
"""
import hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
P=Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/data/amass_g1_36dim/full_train.pkl')
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
sys.path.insert(0,str(R/'ssm_new6_20hz_resume_20261101'))
from g1_fk import URDFFK
NAMES=tuple(n+'_joint' for n in ('left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw'))
TRIPLES=((1,4,7),(2,5,8),(16,18,20),(17,19,21))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def cosine(x):
 out=[]
 for a,b,c in TRIPLES:
  u=x[:,a]-x[:,b];v=x[:,c]-x[:,b];d=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
  if (d<1e-8).any():raise ValueError('degenerate keypoint')
  out.append(np.sum(u*v,axis=1)/d)
 return np.stack(out,axis=1)
def main():
 sh=sha(P)
 if sh!='53a3fff8c2928d3e36b06b1b8bce1148675854ff3c5ef5bad0e56f295abe302d':raise ValueError('corpus hash')
 # This pinned local pickle was acquired earlier from the author's release.
 with P.open('rb') as f:data=pickle.load(f)['data_dict']
 fk=URDFFK(U); dirs=[R/'humanml_mirror_samples_20261102',R/'kit_diverse_clock_samples_20261102'];results=[]
 for directory in dirs:
  for file in sorted(directory.glob('*.npy')):
   sid=file.stem
   if sid not in data or not sid.isdigit():continue
   human=np.load(file,allow_pickle=False);m=np.asarray(data[sid]['motion'],dtype=np.float64)
   if human.ndim!=3 or human.shape[1:]!=(22,3) or m.ndim!=2 or m.shape[1]!=36:raise ValueError('unexpected shape')
   human_ang=cosine(human); robot_pts=fk.forward(m[:,:3],m[:,3:7],m[:,7:],NAMES);robot_ang=cosine(robot_pts)
   hypotheses={}
   for fps in (20.,60.):
    hx=np.arange(len(m))*20./fps;mask=hx<=len(human)-1; n=int(mask.sum()); idx=hx[mask]
    if n<12:continue
    interp=np.column_stack([np.interp(idx,np.arange(len(human)),human_ang[:,i]) for i in range(len(TRIPLES))]); dif=np.abs(robot_ang[mask]-interp)
    hypotheses[str(int(fps))]={'compared_frames':n,'mean_abs_cosine_error':float(dif.mean()),'per_joint_error':dif.mean(axis=0).tolist()}
   results.append({'id':sid,'human_mirror_sha256':sha(file),'human_mirror_frames_at_nominal20':len(human),'g1_frames':len(m),'fps_hypotheses':hypotheses})
 result={'corpus_sha256':sh,'urdf_sha256':sha(U),'results':results,'assumptions':['Human reference is a third-party 20Hz mirror, except official_012314 separately verified elsewhere.','Both signals start at same semantic phase; no time offsets optimized.','Human–robot morphology, joint definition, cropping, and source identity can change scores.'],'training_ready':False}
 out=R/'reactive_humanml_time_hypotheses_20261103.json'
 with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({'samples':len(results),'cases':[{'id':x['id'],'human':x['human_mirror_frames_at_nominal20'],'robot':x['g1_frames'],'err20':x['fps_hypotheses'].get('20',{}).get('mean_abs_cosine_error'),'err60':x['fps_hypotheses'].get('60',{}).get('mean_abs_cosine_error')} for x in results]}))
if __name__=='__main__':main()
