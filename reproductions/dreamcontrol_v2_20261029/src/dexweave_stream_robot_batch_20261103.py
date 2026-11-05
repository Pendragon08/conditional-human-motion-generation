"""Receive G1 references over stdin and build an isolated technical preview.

Uses no human retargeting and no physical-action screening. All captions remain
unreviewed and no preview is admitted to training by this program.
"""
import hashlib,io,json,os,re,shutil,sys
from pathlib import Path
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
batch=os.environ.get('DEXWEAVE_BATCH_ID','batch10')
if not re.fullmatch(r'batch[0-9]+',batch):raise ValueError('invalid batch id')
D=R/('dexweave_stream_robot_'+batch+'_20261103')
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
T=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/TrajGen')
sys.path[:0]=[str(R),str(R/'ssm_new6_20hz_resume_20261101'),str(R/'antialias_v3_20261030/src'),str(T)]
from g1_fk import URDFFK
from feature263_signfix import encode_signfixed
from resample_antialias_omniretarget import resample_antialias
def digest(b):return hashlib.sha256(b).hexdigest()
def exact(f,n):
 data=bytearray()
 while len(data)<n:
  b=f.read(n-len(data))
  if not b:raise EOFError('truncated frame')
  data.extend(b)
 return bytes(data)
def main():
 if digest(U.read_bytes())!='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95':raise ValueError('URDF changed')
 if shutil.disk_usage(R).free<8*1024**3+256*1024**2+100*1024**2:raise RuntimeError('disk reserve')
 D.mkdir(exist_ok=False); fk=URDFFK(U);names=tuple(fk.actuated); n=0
 with (D/'records.jsonl').open('x',encoding='utf8') as log:
  while True:
   line=sys.stdin.buffer.readline()
   if not line:break
   h=json.loads(line); length=int(h['bytes'])
   if not 0<length<=30000000:raise ValueError('input size')
   raw=exact(sys.stdin.buffer,length)
   if digest(raw)!=h['published_sha256']:raise ValueError('published hash')
   if shutil.disk_usage(R).free<8*1024**3+256*1024**2+100*1024**2:raise RuntimeError('disk reserve')
   record={'id':h['id'],'input_sha256':digest(raw),'source_member_sha256':h['source_member_sha256'],'text_sha256':h['text_sha256'],'captions':h['captions'],'risk_codes':h['risk_codes'],'caption_accepted':False,'training_ready':False}
   try:
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
     sn=list(f['joint_names']);src=f['source_seq_name'].item();sh=f['source_motion_sha256'].item();fps=float(f['fps']);map_=f['retime_source_indices'];full=int(f['source_frame_count']); q=f['joint_pos'][:,[sn.index(x) for x in names]];pos=f['root_pos'];rot=f['root_quat'];obj=f['object_pos'];oq=f['object_quat'];on=f['object_names']
    if len(names)!=29 or len(set(sn))!=len(sn) or src!=h['id'] or sh!=h['source_member_sha256'] or fps!=50 or full!=h['source_frames'] or len(map_)!=len(q) or np.any(np.diff(map_)<0) or map_[0]<0 or map_[-1]>full-1:raise ValueError('source identity or time metadata')
    qpos=np.column_stack((rot,pos,q));sampled=resample_antialias(qpos,fps,20.,max_seconds=10.);points=fk.forward(sampled[:,4:7],sampled[:,[1,2,3,0]],sampled[:,7:],names);features=encode_signfixed(points,foot_velocity_squared_threshold=.002)
    if features.shape!=(len(sampled)-1,263) or not np.isfinite(features).all():raise ValueError('features')
    o20=np.stack([resample_antialias(np.column_stack((qpos,oq[:,i],obj[:,i])),fps,20.,max_seconds=10.)[:,36:43] for i in range(obj.shape[1])],axis=1)
    target=D/('sample_%04d.npz'%n)
    with target.open('xb') as f:np.savez_compressed(f,state29_wxyz_xyz=sampled,points_z_up=points,features=features,joint_names=np.asarray(names),object_pose_wxyz_xyz=o20,object_names=on,sample_times=np.arange(len(sampled))/20.)
    record.update(status='technical_preview',artifact=target.name,artifact_sha256=digest(target.read_bytes()),states=len(sampled),feature_frames=len(features),source_fps=fps,source_frame_count=full,source_index_first=float(map_[0]),source_index_last=float(map_[-1]),cropped_10_seconds=bool((len(q)-1)/fps>=10.))
   except Exception as e:record.update(status='technical_error',error=type(e).__name__+': '+str(e))
   log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');log.flush();n+=1
   print(json.dumps({'id':h['id'],'status':record['status'],'index':n},ensure_ascii=False),flush=True)
 with (D/'report.json').open('x',encoding='utf8') as f:json.dump({'count':n,'training_ready':False,'physical_action_filter':False,'note':'Technical previews only; captions and model geometry not accepted.'},f,indent=2)
if __name__=='__main__':main()
