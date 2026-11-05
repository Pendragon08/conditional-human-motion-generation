"""Seven-frame GMR-initialized GPU PyRoki smoke, not a physical acceptance."""
import hashlib,json,sys,time
from pathlib import Path
import jax,jax.numpy as jnp,jaxlie,numpy as np,pyroki as pk,yourdfpy
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
P=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl')
H=R/'gmr_pyroki_initialization_20261103';sys.path.insert(0,str(H))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,create_conn_tree,solve_retargeting,foot_detect
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise RuntimeError('one dedicated visible GPU required')
 source=R/'gmr_grab_locked27_smoke_v2_20261103/gmr_source_rate.npz'
 rec=json.loads((source.parent/'report.json').read_text());helper=H/'retarget_helpers/_utils.py';hp=json.loads((H/'provenance.json').read_text())
 urdf=P/'Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf'
 receipt=json.loads((R/'pyroki_grab_temporal_20261030/report.json').read_text())
 if sha(source)!=rec['output_sha256'] or sha(helper)!=hp['helper_sha256'] or sha(urdf)!=receipt['urdf_sha256']:raise ValueError('hash mismatch')
 with np.load(source,allow_pickle=False) as f:
  human=f['human_joints'][:8].copy();root=f['root_wxyz_xyz'][1:8].copy();q=f['joint_pos'][1:8].copy();names=list(f['joint_names']);frames=f['source_frames'][1:8].copy();fps=float(f['declared_fps'])
 urdf_data=yourdfpy.URDF.load(str(urdf),load_meshes=False,load_collision_meshes=True)
 robot=pk.Robot.from_urdf(urdf_data);collision=pk.collision.RobotCollision.from_urdf(urdf_data);target_names=list(robot.joints.actuated_names)
 if set(names)!=set(target_names) or len(names)!=27:raise ValueError('joint names')
 q=q[:,[names.index(n) for n in target_names]]
 hi,ri=get_humanoid_retarget_indices_g1_27dof();mask=create_conn_tree(robot,ri)
 plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),height_data=jnp.zeros((100,100),dtype=jnp.float32))
 left,right=foot_detect(human[None],thres=(.05/fps)**2);target=jnp.asarray(human[1:]);weights=dict(receipt['weights']);weights['root_smoothness']=weights['foot_skating']=2*fps/20
 began=time.monotonic()
 result_root,result_q=solve_retargeting(robot=robot,robot_coll=collision,target_keypoints=target,is_left_foot_contact=jnp.asarray(left[0]>0),is_right_foot_contact=jnp.asarray(right[0]>0),left_foot_keypoints=plane.project_points(target[:,10,:]),right_foot_keypoints=plane.project_points(target[:,11,:]),smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=mask,heightmap=plane,weights=weights,pick_point=jnp.zeros(3),gmr_root=jnp.asarray(root),gmr_joints=jnp.asarray(q))
 result_root=np.asarray(result_root.wxyz_xyz);result_q=np.asarray(result_q)
 if result_root.shape!=(7,7) or result_q.shape!=(7,27) or not np.isfinite(result_root).all() or not np.isfinite(result_q).all():raise ValueError('invalid result')
 out=R/'gmr_pyroki_gpu_smoke_20261103';out.mkdir(exist_ok=False)
 with (out/'result.npz').open('xb') as f:np.savez_compressed(f,source_frames=frames,root_wxyz_xyz=result_root,joint_pos=result_q,joint_names=np.asarray(target_names),gmr_root_wxyz_xyz=root,gmr_joint_pos=q)
 report={'source_sha256':sha(source),'helper_sha256':sha(helper),'urdf_sha256':sha(urdf),'output_sha256':sha(out/'result.npz'),'frames':len(frames),'backend':str(jax.devices()[0]),'elapsed_s':time.monotonic()-began,'max_joint_change_rad':float(np.max(np.abs(result_q-q))),'training_ready':False,'limitations':['Seven frame isolated warm-start smoke only.','GMR model and target URDF differ; source clock, contact, physical tracking and text remain unverified.','Legacy objective and 120Hz weight scaling are approximations, not author v2.']}
 with (out/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
