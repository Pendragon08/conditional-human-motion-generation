"""Read-only CPU check of the actual native MotionLib on new mixed inputs.

No simulator, optimizer or training loop is started. Independently checks CSV
conversion, source20Hz -> native50Hz time grids, all clips' future references,
and native FK against MuJoCo at each clip's start/middle/end.
"""
import argparse,hashlib,json,os,pathlib,sys,time,traceback
import numpy as np
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
N=R/'stage3_supermimic29_20261103/native_universal_20261104'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def atomic(p,j):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(p)
def quat_matrix(q):
 q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=q.T
 return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)
def slerp(q0,q1,a):
 q0=q0.astype(float);q1=q1.astype(float);q0/=np.linalg.norm(q0,axis=-1,keepdims=True);q1/=np.linalg.norm(q1,axis=-1,keepdims=True)
 dot=np.sum(q0*q1,axis=-1);q1=np.where((dot<0)[:,None],-q1,q1);dot=np.clip(np.abs(dot),0,1)
 angle=np.arccos(dot);sine=np.sin(angle);safe=np.maximum(sine,1e-12)
 curved=(np.sin((1-a)*angle)/safe)[:,None]*q0+(np.sin(a*angle)/safe)[:,None]*q1
 out=np.where((dot>.9995)[:,None],q0*(1-a[:,None])+q1*a[:,None],curved)
 return out/np.linalg.norm(out,axis=-1,keepdims=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared-dir',type=pathlib.Path,required=True);p.add_argument('--output-dir',type=pathlib.Path,required=True);a=p.parse_args()
 assert not a.output_dir.exists();a.output_dir.mkdir(exist_ok=False);began=time.monotonic()
 state={'status':'initializing','pid':os.getpid(),'device':'cpu','optimizer_updates':0,'formal_training_started':False};atomic(a.output_dir/'status.json',state)
 try:
  prepared=json.loads((a.prepared_dir/'status.json').read_text());manifest_path=a.prepared_dir/'native_motion_manifest.json';m=json.loads(manifest_path.read_text())
  assert sha(manifest_path)==prepared['manifest_sha256'] and m['reference_count']==(8 if m['preview'] else 2000)
  assert m['per_task']=={t:2 if m['preview'] else 500 for t in ['pick','pour','bimanual_box_lift','box_push_pull']}
  spec=json.loads((N/'environment_spec.json').read_text());assert spec['spec_hash']=='32a40546';sys.path[:0]=spec['spec']['module_paths']
  import torch,mujoco
  torch.set_num_threads(2)
  from supermimic.motion import MotionLib,CONTROL_DT
  from supermimic.kinematics import G1Kinematics
  from supermimic.robot import G1_JOINT_NAMES,MJCF_PATH,TRACKED_BODIES
  assert G1_JOINT_NAMES==m['body_joint_order'] and CONTROL_DT==.02
  source_bindings={str(N/'source/supermimic'/f):sha(N/'source/supermimic'/f) for f in ['motion.py','kinematics.py','robot.py','rotations.py']}
  manifest_records=sorted(m['records'],key=lambda r:pathlib.Path(r['csv']).name);files=[r['csv'] for r in manifest_records]
  assert files==sorted(str(x) for x in (a.prepared_dir/'motion_csv20').glob('*.csv'))
  for r in manifest_records:assert sha(r['csv'])==r['csv_sha256'] and sha(r['source'])==r['source_sha256']
  state.update(status='native_motionlib_cpu_loading',reference_count=len(files));atomic(a.output_dir/'status.json',state)
  kin=G1Kinematics('cpu');motion=MotionLib.from_dir(str(a.prepared_dir/'motion_csv20'),kin,'cpu')
  assert motion.clip_len.numel()==len(files) and motion.q.shape==(motion.num_frames,29)
  assert motion.names==[pathlib.Path(f).stem for f in files]
  for key in ['root_p','root_R','root_v','root_w','q','qd','body_p','body_R','body_v','body_w']:
   assert bool(torch.isfinite(getattr(motion,key)).all()),key
  model=mujoco.MjModel.from_xml_path(MJCF_PATH);data=mujoco.MjData(model)
  joint_addresses=[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in G1_JOINT_NAMES]
  free=int(model.jnt_qposadr[next(i for i,x in enumerate(model.jnt_type) if x==mujoco.mjtJoint.mjJNT_FREE)])
  body_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in TRACKED_BODIES];assert min(body_ids)>0
  details=[];total_expected=0;max_csv=0.;max_q=0.;max_root=0.;max_R=0.;max_fk=0.;max_fk_R=0.
  for c,r in enumerate(manifest_records):
   raw=np.loadtxt(r['csv'],delimiter=',')
   with np.load(r['source'],allow_pickle=False) as z:
    order=[z['joint_names'].tolist().index(n) for n in G1_JOINT_NAMES]
    expected=np.c_[z['root_pos'],z['root_quat'][:,[1,2,3,0]],z['q'][:,order]]
    assert raw.shape==(r['source_frames'],36) and np.array_equal(z['sample_times'],np.arange(r['source_frames'])/20)
    csv_error=float(np.max(np.abs(raw-expected)));assert csv_error<1e-7
   # Native uses an inclusive grid ending at the last available50Hz tick,
   # never stretches source120/180 samples to6/9 full seconds.
   duration=(len(raw)-1)/20.;length=int(np.floor(duration/.02))+1
   s=int(motion.clip_start[c]);e=int(motion.clip_end[c]);assert int(motion.clip_len[c])==length and e==s+length-1
   assert motion.clip_id[s:e+1].eq(c).all() and (length-1)*.02<=duration+1e-10
   source_times=np.arange(len(raw))/20.;target_times=np.arange(length)*.02
   positions=np.column_stack([np.interp(target_times,source_times,raw[:,k]) for k in range(3)])
   joints=np.column_stack([np.interp(target_times,source_times,raw[:,k]) for k in range(7,36)])
   u=target_times*20.;i0=np.minimum(u.astype(int),len(raw)-2);alpha=np.clip(u-i0,0,1)
   quats=slerp(raw[i0][:,[6,3,4,5]],raw[i0+1][:,[6,3,4,5]],alpha);rotations=quat_matrix(quats)
   q_error=float(np.max(np.abs(joints-motion.q[s:e+1].numpy())));root_error=float(np.max(np.abs(positions-motion.root_p[s:e+1].numpy())))
   rotation_error=float(np.max(np.abs(rotations-motion.root_R[s:e+1].numpy())))
   assert q_error<2e-5 and root_error<2e-5 and rotation_error<2e-5
   for local in [0,length//2,length-1]:
    data.qpos[:]=model.qpos0;data.qpos[free:free+3]=positions[local];data.qpos[free+3:free+7]=quats[local]
    data.qpos[joint_addresses]=joints[local];mujoco.mj_forward(model,data)
    fk_error=float(np.max(np.abs(data.xpos[body_ids]-motion.body_p[s+local].numpy())))
    fk_R_error=float(np.max(np.abs(data.xmat[body_ids].reshape(-1,3,3)-motion.body_R[s+local].numpy())))
    assert fk_error<3e-5 and fk_R_error<3e-5
    max_fk=max(max_fk,fk_error);max_fk_R=max(max_fk_R,fk_R_error)
   frames=torch.tensor([s,e-5,e],dtype=torch.long);future=motion.future_frames(frames,10);goal=motion.goal(frames,10).reshape(3,10,73)
   assert goal.shape==(3,10,73) and torch.isfinite(goal).all() and motion.clip_id[future].eq(c).all()
   assert torch.equal(goal[:,:,15:44],motion.q[future])
   requested=frames[:,None]+torch.arange(1,11);extension=requested>e
   assert torch.count_nonzero(goal[:,:,9:15][extension])==0 and torch.count_nonzero(goal[:,:,44:73][extension])==0
   assert torch.count_nonzero(goal[-1,:,:3])==0 and future[-1].eq(e).all()
   max_csv=max(max_csv,csv_error);max_q=max(max_q,q_error);max_root=max(max_root,root_error);max_R=max(max_R,rotation_error);total_expected+=length
   details.append({'task':r['task'],'candidate_id':r['candidate_id'],'native_clip':c,'native_length':length,'source_frames':len(raw),
    'source_last_time_s':duration,'native_last_time_s':float(target_times[-1]),'csv_conversion_max_error':csv_error,'q_resample_max_error':q_error,
    'root_resample_max_error':root_error,'rotation_resample_max_error':rotation_error,'future_reference_boundary_passed':True})
   if (c+1)%50==0:
    state.update(status='checking_all_native_clips',clips_checked=c+1,elapsed_seconds=time.monotonic()-began);atomic(a.output_dir/'status.json',state)
  assert total_expected==motion.num_frames and sha(manifest_path)==prepared['manifest_sha256']
  result={'status':'preview_native_library_passed' if m['preview'] else 'all2000_native_library_passed','preview':m['preview'],
   'reference_count':len(files),'per_task':m['per_task'],'manifest':str(manifest_path),'manifest_sha256':sha(manifest_path),
   'native50Hz_frames':motion.num_frames,'native_body_dof':29,'source20Hz_preserved':True,'all_clips_future_goal_and_boundary_passed':True,
   'mujoco_fk_samples':3*len(files),'max_csv_conversion_error':max_csv,'max_joint_resample_error':max_q,'max_root_resample_error':max_root,
   'max_rotation_resample_error':max_R,'max_native_fk_vs_mujoco_position_error':max_fk,'max_native_fk_vs_mujoco_rotation_error':max_fk_R,
   'source_bindings':source_bindings,'mjcf':MJCF_PATH,'mjcf_sha256':sha(MJCF_PATH),'environment_spec_hash':spec['spec_hash'],
   'worker_sha256':sha(__file__),'optimizer_updates':0,'formal_training_started':False,'device':'cpu','details':details,
   'elapsed_seconds':time.monotonic()-began,'scope':'actual upstream MotionLib loading and independent data/FK/goal checks, no policy or object-interaction performance claim'}
  (a.output_dir/'report.json').open('x').write(json.dumps(result,indent=2,allow_nan=False))
  state.update({k:v for k,v in result.items() if k!='details'});state['report_sha256']=sha(a.output_dir/'report.json');atomic(a.output_dir/'status.json',state)
  print(json.dumps({k:v for k,v in state.items() if k!='source_bindings'}),flush=True)
 except BaseException as e:
  state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());atomic(a.output_dir/'status.json',state);raise

if __name__=='__main__':main()
