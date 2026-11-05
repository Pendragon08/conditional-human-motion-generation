"""One hash-pinned, non-training-ready DexWeave + MotionHub pilot.

Human-to-robot retargeting and physical-action filtering are not performed.
"""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'dexweave_grab_pilot_20261103'
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path[:0]=[str(R),str(R/'ssm_new6_20hz_resume_20261101'),str(R/'antialias_v3_20261030/src'),str(T)]
from g1_fk import URDFFK
from feature263_signfix import encode_signfixed
from resample_antialias_omniretarget import resample_antialias
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if shutil.disk_usage(R).free<8*1024**3+256*1024**2:raise RuntimeError('disk reserve')
 motion=D/'wineglass_pass_1_opt.npz'; text=D/'motionhub_s9_wineglass_pass_1.json'
 if sha(motion)!='f0f7df22e89e2996a8d88fc5e961f38d4955e5795c3478a1d804629922dd664e' or sha(text)!='c9d4b35cd485642a3b1e88feac122d6b55f869779b38e0ef4dcf565b92f1db6a':raise ValueError('sample hash')
 if sha(U)!='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95':raise ValueError('URDF hash')
 fk=URDFFK(U);names=tuple(fk.actuated)
 with np.load(motion,allow_pickle=False) as f:
  all_names=list(f['joint_names']);idx=[all_names.index(n) for n in names]; q=f['joint_pos'][:,idx]
  root=f['root_pos'];rot=f['root_quat'];fps=float(f['fps']);src=f['source_seq_name'].item();source_hash=f['source_motion_sha256'].item();mapping=f['retime_source_indices']
  obj=f['object_pos'];oq=f['object_quat'];on=f['object_names']
 if len(names)!=29 or len(set(all_names))!=53 or src!='s9/wineglass_pass_1' or source_hash!='8cd85e96692e2a27bbe203d1e0b3e38fde04a28a645d56df330ad66c8d294210' or fps!=50 or len(mapping)!=len(q):raise ValueError('source contract')
 if not np.all(np.diff(mapping)>=0) or mapping[0]<0 or mapping[-1]>799:raise ValueError('source frame mapping')
 qpos=np.column_stack((rot,root,q));sampled=resample_antialias(qpos,fps,20.,max_seconds=10.)
 points=fk.forward(sampled[:,4:7],sampled[:,[1,2,3,0]],sampled[:,7:],names)
 features=encode_signfixed(points,foot_velocity_squared_threshold=.002)
 if features.shape!=(len(sampled)-1,263) or not np.isfinite(features).all():raise ValueError('features')
 object20=[]
 for i in range(obj.shape[1]):
  combined=np.column_stack((qpos,oq[:,i],obj[:,i]))
  object20.append(resample_antialias(combined,fps,20.,max_seconds=10.)[:,36:43])
 object20=np.stack(object20,axis=1)
 out=D/'preview.npz'
 with out.open('xb') as f:np.savez_compressed(f,state29_wxyz_xyz=sampled,points_z_up=points,features=features,joint_names=np.asarray(names),object_pose_wxyz_xyz=object20,object_names=on,sample_times=np.arange(len(sampled))/20.)
 r={'id':src,'source_member_sha256':source_hash,'motion_sha256':sha(motion),'caption_sha256':sha(text),'urdf_sha256':sha(U),'source_frames':len(q),'source_fps':fps,'output_fps':20,'output_states':len(sampled),'output_feature_frames':len(features),'object_count':obj.shape[1],'output_sha256':sha(out),'caption_accepted':False,'training_ready':False,'limitations':['Publisher source hash matches local GRAB registry, not an independent extraction in this run.','Different robot URDF has not been compared link by link.','Nominal sampled clock, semantics, caption and object sync not accepted.','Quaternion component filtering is approximate.','No physical action filtering applied.']}
 with (D/'preview_report.json').open('x',encoding='utf8') as f:json.dump(r,f,indent=2,allow_nan=False)
 print(json.dumps(r))
if __name__=='__main__':main()
