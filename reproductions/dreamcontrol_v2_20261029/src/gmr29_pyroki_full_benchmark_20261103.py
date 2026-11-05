"""Full-sequence GMR-initialized PyRoki windows. Exploratory 29DoF benchmark.

Each saved window is provisional: overlap and physical/text quality must be
evaluated before stitching or training. Never modifies source or vendor assets.
"""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import jax,jax.numpy as jnp,jaxlie,numpy as np,pyroki as pk,yourdfpy
from scipy.spatial.transform import Rotation
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
P=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
H=R/'gmr_pyroki_initialization_20261103';sys.path.insert(0,str(H))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,G1_27_dof,create_conn_tree,solve_retargeting,foot_detect
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 if shutil.disk_usage(R).free < 8*1024**3+256*1024**2:raise RuntimeError('disk reserve guard')
 line=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
 free,total=map(lambda x:int(x.strip()),line)
 if free<2048:raise RuntimeError('shared gpu memory guard')
 # Stop between windows when another compute process is using the shared GPU.
 activity=subprocess.check_output(['nvidia-smi','pmon','-i','1','-c','1','-s','u'],text=True,timeout=8)
 for line in activity.splitlines():
  fields=line.split()
  if len(fields)<5 or fields[0].startswith('#') or fields[2]!='C':continue
  pid=int(fields[1]);sm=0 if fields[3]=='-' else int(fields[3])
  if pid!=os.getpid() and sm>=10:raise RuntimeError('another GPU compute process is active')
def main():
 guard();devices=jax.devices()
 if len(devices)!=1 or devices[0].platform!='gpu':raise RuntimeError('one visible GPU required')
 src=R/'gmr_grab29_full_airplane_20261103/gmr_source_rate.npz';rec=json.loads((src.parent/'report.json').read_text())
 helper=H/'retarget_helpers/_utils.py';hp=json.loads((H/'provenance.json').read_text());urdf=P/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf'
 receipt=json.loads((R/'pyroki_grab_temporal_20261030/report.json').read_text())
 if sha(src)!=rec['output_sha256'] or sha(helper)!=hp['helper_sha256'] or sha(urdf)!='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95':raise ValueError('hash mismatch')
 with np.load(src,allow_pickle=False) as f:
  human=f['human_joints'].copy();groot=f['root_wxyz_xyz'].copy();gq=f['joint_pos'].copy();names=list(f['joint_names']);frames=f['source_frames'].copy();fps=float(f['declared_fps'])
 if len(frames)<8 or not np.array_equal(np.diff(frames),np.ones(len(frames)-1,dtype=int)) or len(names)!=29:raise ValueError('invalid source')
 model=yourdfpy.URDF.load(str(urdf),load_meshes=False,load_collision_meshes=True);robot=pk.Robot.from_urdf(model);collision=pk.collision.RobotCollision.from_urdf(model);target_names=list(robot.joints.actuated_names)
 if set(names)!=set(target_names):raise ValueError('joint mismatch')
 gq=gq[:,[names.index(n) for n in target_names]]
 hi,oldri=get_humanoid_retarget_indices_g1_27dof();ri=jnp.asarray([robot.links.names.index(G1_27_dof[int(i)]) for i in np.asarray(oldri)]);mask=create_conn_tree(robot,ri)
 plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),height_data=jnp.zeros((100,100),dtype=jnp.float32))
 weights=dict(receipt['weights']);weights['root_smoothness']=weights['foot_skating']=2*fps/20
 starts=list(range(0,len(frames)-7,4))
 if starts[-1]!=len(frames)-8:starts.append(len(frames)-8)
 out=R/'gmr29_pyroki_full_airplane_20261103';out.mkdir(exist_ok=False);started=time.monotonic();previous=None;durations=[]
 with (out/'windows.jsonl').open('x',buffering=1) as log:
  for wi,start in enumerate(starts):
   if wi%5==0:guard()
   h=human[start:start+8];target=jnp.asarray(h[1:]);left,right=foot_detect(h[None],thres=(.05/fps)**2);t=time.monotonic()
   root,q=solve_retargeting(robot=robot,robot_coll=collision,target_keypoints=target,is_left_foot_contact=jnp.asarray(left[0]>0),is_right_foot_contact=jnp.asarray(right[0]>0),left_foot_keypoints=plane.project_points(target[:,10,:]),right_foot_keypoints=plane.project_points(target[:,11,:]),smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,heightmap=plane,weights=weights,pick_point=jnp.zeros(3),gmr_root=jnp.asarray(groot[start+1:start+8]),gmr_joints=jnp.asarray(gq[start+1:start+8]))
   root=np.asarray(root.wxyz_xyz);q=np.asarray(q);elapsed=time.monotonic()-t;durations.append(elapsed);idx=np.arange(start+1,start+8)
   if root.shape!=(7,7) or q.shape!=(7,29) or not np.isfinite(root).all() or not np.isfinite(q).all():raise ValueError('nonfinite solver output')
   overlap=None
   if previous is not None:
    common=np.intersect1d(previous['indices'],idx);a=np.searchsorted(previous['indices'],common);b=np.searchsorted(idx,common)
    qa=Rotation.from_quat(previous['root'][a][:,[1,2,3,0]]);qb=Rotation.from_quat(root[b][:,[1,2,3,0]])
    overlap={'frames':common.tolist(),'root_m':float(np.max(np.linalg.norm(previous['root'][a,4:]-root[b,4:],axis=1))),'root_angle_rad':float(np.max((qa.inv()*qb).magnitude())),'joint_rad':float(np.max(np.abs(previous['q'][a]-q[b])))}
   p=out/f'window_{wi:04d}.npz'
   with p.open('xb') as f:np.savez_compressed(f,indices=idx,source_frames=frames[idx],joint_names=np.asarray(target_names),root_wxyz_xyz=root,joint_pos=q)
   row={'window':wi,'start':start,'source_frames':frames[idx].tolist(),'file':p.name,'sha256':sha(p),'elapsed_s':elapsed,'overlap':overlap}
   log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno());previous={'indices':idx,'root':root,'q':q}
   if (wi+1)%10==0:print('windows',wi+1,'of',len(starts),flush=True)
 if sha(src)!=rec['output_sha256']:raise RuntimeError('input changed')
 report={'source_sha256':sha(src),'helper_sha256':sha(helper),'urdf_sha256':sha(urdf),'rows_sha256':sha(out/'windows.jsonl'),'source_frames':len(frames),'window_count':len(starts),'elapsed_s':time.monotonic()-started,'first_window_s':durations[0],'subsequent_median_window_s':float(np.median(durations[1:])),'backend':str(devices[0]),'training_ready':False,'limitations':['Source clock is declared metadata.','8-frame windows with 4-frame stride; output not stitched or accepted.','Prior objective and source-rate temporal weights are approximations; model FK differs from GMR asset.','Caption/physical tracking are not checked.']}
 with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
