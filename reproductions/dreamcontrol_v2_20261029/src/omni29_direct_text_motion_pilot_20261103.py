"""Direct official 29DoF Omni trajectory + original text -> 263D pilot.

No human/robot retargeting, no physical filtering. Semantic match and source
rights must be reviewed; this artifact is not a training-ready dataset.
"""
import hashlib,io,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path[:0]=[str(R),str(R/'ssm_new6_20hz_resume_20261101'),str(R/'antialias_v3_20261030/src'),str(T)]
from g1_fk import URDFFK
from feature263_signfix import encode_signfixed
from resample_antialias_omniretarget import resample_antialias
names=tuple(n+'_joint' for n in ('left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 if shutil.disk_usage(R).free<8*1024**3+256*1024**2:raise RuntimeError('disk reserve')
 archive=R/'data/OmniRetarget_Dataset/robot-object.zip';textzip=R/'omomo_text_pairing_20261101/omomo_text_anno.zip'
 if sha(archive)!='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3' or sha(textzip)!='2d11fb37fef294b7f6816c74740694d12f28d62e0f638e348c678dc51ee1ad7c' or sha(U)!='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95':raise ValueError('pinned input changed')
 member='robot-object/sub3_largebox_011_original.npz';annotation='omomo_text_anno_json_data/sub3_largebox_011.json'
 with zipfile.ZipFile(archive) as z:raw=z.read(member)
 with zipfile.ZipFile(textzip) as z:annraw=z.read(annotation)
 if hashlib.sha256(raw).hexdigest()!='d8024a424d51ed7e8edd55bdc8392fcdd8971ac0216020df92f79ce2236890f9' or hashlib.sha256(annraw).hexdigest()!='fc33352664cd4dd8c9be451a5af91c3b6ef9637ae38229b75fc4bb114b5ce5c5':raise ValueError('member mismatch')
 caption=json.loads(annraw)['sub3_largebox_011']
 with np.load(io.BytesIO(raw),allow_pickle=False) as f:q=np.asarray(f['qpos']);fps=float(f['fps'])
 if q.shape!=(192,43) or fps!=30:raise ValueError('unexpected source')
 sampled=resample_antialias(q,fps,20.,max_seconds=10.)
 fk=URDFFK(U)
 if set(fk.actuated)!=set(names):raise ValueError('joint names')
 points=fk.forward(sampled[:,4:7],sampled[:,[1,2,3,0]],sampled[:,7:36],names)
 features=encode_signfixed(points,foot_velocity_squared_threshold=.002)
 if features.shape!=(len(sampled)-1,263):raise ValueError('invalid features')
 out=R/'omni29_direct_text_motion_pilot_20261103';out.mkdir(exist_ok=False)
 with (out/'state_features.npz').open('xb') as f:np.savez_compressed(f,qpos29_source_layout_20hz=sampled,points_z_up=points,features=features,sample_times=np.arange(len(sampled))/20.,joint_names=np.asarray(names))
 report={'archive_sha256':sha(archive),'source_member':member,'source_member_sha256':hashlib.sha256(raw).hexdigest(),'annotation_archive_sha256':sha(textzip),'annotation_member':annotation,'annotation_sha256':hashlib.sha256(annraw).hexdigest(),'caption':caption,'caption_status':'nominal_source_match_with_prior_limited_human_video_review_not_accepted','urdf_sha256':sha(U),'source_fps_declared':fps,'target_fps':20,'states':len(sampled),'feature_frames':len(features),'output_sha256':sha(out/'state_features.npz'),'training_ready':False,'limitations':['Original 29DoF trajectory directly adapted, not retargeted.','Anti-alias component quaternion filtering and sign-fixed HumanML encoding are approximations, not author code.','Nominal caption association and previous human source review do not establish 29DoF motion semantics or identity.','Capture clock, license scope and training acceptance unresolved; no physical filtering applied.']}
 with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
