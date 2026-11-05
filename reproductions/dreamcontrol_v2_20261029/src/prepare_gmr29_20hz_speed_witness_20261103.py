"""Exploratory 120Hz->20Hz pre-optimization witness, not paper preprocessing.

The input GMR result itself was computed at full source rate. This measures a
possible PyRoki-stage shortcut only; no new training data is accepted.
"""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(R/'antialias_v3_20261030/src'))
from resample_antialias_omniretarget import resample_antialias
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def linear_channels_120_to_20(values):
 x=np.asarray(values,dtype=np.float64);shape=x.shape;z=x.reshape(len(x),-1)
 line=z[0]+np.arange(len(z))[:,None]/(len(z)-1)*(z[-1]-z[0]);res=resample_poly(z-line,1,6,axis=0,window=('kaiser',5.0),padtype='line')
 res+=z[0]+np.arange(len(res))[:,None]*6/(len(z)-1)*(z[-1]-z[0]);res[0]=z[0]
 return res.reshape((len(res),)+shape[1:])
def main():
 if shutil.disk_usage(R).free<8*1024**3+256*1024**2:raise RuntimeError('disk reserve')
 source=R/'gmr_grab29_full_airplane_20261103/gmr_source_rate.npz';rep=json.loads((source.parent/'report.json').read_text())
 if sha(source)!=rep['output_sha256']:raise ValueError('source hash')
 with np.load(source,allow_pickle=False) as f:
  frames=f['source_frames'];fps=float(f['declared_fps']);root=f['root_wxyz_xyz'];joints=f['joint_pos'];human=f['human_joints'];names=f['joint_names'];opos=f['object_transl'];orot=f['object_orient_rotvec']
 if len(frames)!=1198 or fps!=120 or not np.array_equal(frames,np.arange(1198)):raise ValueError('source timeline')
 objquat=Rotation.from_rotvec(orot).as_quat()[:,[3,0,1,2]]
 source_qpos=np.concatenate([root,joints,objquat,opos],axis=1)
 result=resample_antialias(source_qpos,120.,20.);human20=linear_channels_120_to_20(human)
 if len(result)!=len(human20) or len(result)!=200:raise ValueError('resampling coverage')
 sample_frames=np.arange(len(result))*6
 out=R/'gmr29_20hz_speed_witness_20261103';out.mkdir(exist_ok=False);p=out/'state_20hz.npz'
 with p.open('xb') as f:np.savez_compressed(f,source_frames=sample_frames,source_declared_fps=np.array(120.),processed_fps=np.array(20.),root_wxyz_xyz=result[:,:7],joint_pos=result[:,7:36],joint_names=names,human_joints=human20,object_wxyz_xyz=result[:,36:43])
 report={'source_sha256':sha(source),'output_sha256':sha(p),'source_frames':len(frames),'processed_frames':len(result),'source_declared_fps':120.,'processed_fps':20.,'source_frame_first_last':[int(sample_frames[0]),int(sample_frames[-1])],'training_ready':False,'limitations':['Resampling before PyRoki is an explicit speed experiment differing from paper processing order.','GMR still ran at 120Hz: no GMR-stage speedup measured.','Component-filtered quaternion normalization approximates SO(3) filtering.','No comparison of optimized trajectory, contact or captions; no physical-action filter.']}
 with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps(report))
if __name__=='__main__':main()
