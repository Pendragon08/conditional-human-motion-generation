"""One isolated overlap-anchor witness; not a physical or training decision."""
import hashlib,json,sys,time
from pathlib import Path
import jax,jax.numpy as jnp,jaxlie,numpy as np,pyroki as pk,yourdfpy
from scipy.spatial.transform import Rotation
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');H=R/'gmr29_anchor_limit_v2_helper_20261103';sys.path.insert(0,str(H))
from retarget_helpers._utils import get_humanoid_retarget_indices_g1_27dof,G1_27_dof,create_conn_tree,solve_retargeting,foot_detect
U=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if len(jax.devices())!=1 or jax.devices()[0].platform!='gpu':raise ValueError('explicit GPU required')
 source=R/'gmr_grab29_full_airplane_20261103/gmr_source_rate.npz';sr=json.loads((source.parent/'report.json').read_text());prior=R/'gmr29_pyroki_full_airplane_20261103/window_0135.npz';prior_next=prior.with_name('window_0136.npz')
 h=H/'retarget_helpers/_utils.py';hr=json.loads((H/'provenance.json').read_text())
 if sha(source)!=sr['output_sha256'] or sha(h)!=hr['helper_sha256'] or sha(U)!='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95':raise ValueError('input hash mismatch')
 manifest=prior.parent/'windows.jsonl';baseline_report=json.loads((prior.parent/'report.json').read_text())
 if sha(manifest)!=baseline_report['rows_sha256']:raise ValueError('baseline manifest hash')
 rows=[json.loads(x) for x in manifest.read_text().splitlines()]
 if rows[135]['sha256']!=sha(prior) or rows[136]['sha256']!=sha(prior_next):raise ValueError('baseline window hash')
 with np.load(source,allow_pickle=False) as f:human=f['human_joints'][544:552].copy();q=f['joint_pos'][545:552].copy();root=f['root_wxyz_xyz'][545:552].copy();names=list(f['joint_names']);fps=float(f['declared_fps'])
 with np.load(prior,allow_pickle=False) as f:pi=f['indices'];pn=list(f['joint_names']);pq=f['joint_pos'];pr=f['root_wxyz_xyz']
 with np.load(prior_next,allow_pickle=False) as f:bi=f['indices'];bq=f['joint_pos'];br=f['root_wxyz_xyz'];bn=list(f['joint_names'])
 model=yourdfpy.URDF.load(str(U),load_meshes=False,load_collision_meshes=True);robot=pk.Robot.from_urdf(model);coll=pk.collision.RobotCollision.from_urdf(model);target_names=list(robot.joints.actuated_names)
 if set(names)!=set(target_names) or pn!=target_names or bn!=target_names:raise ValueError('joint order mismatch')
 q=q[:,[names.index(n) for n in target_names]];idx=np.arange(545,552);common,pa,na=np.intersect1d(pi,idx,return_indices=True)
 if not np.array_equal(common,[545,546,547]) or not np.array_equal(bi,idx):raise ValueError('overlap mismatch')
 anchor_root=root.copy();anchor_q=q.copy();mask=np.zeros(7,dtype=np.float32);anchor_root[na]=pr[pa];anchor_q[na]=pq[pa];mask[na]=1
 # Initialize the overlap from prior output and retain GMR initialization elsewhere.
 initial_root=root.copy();initial_q=q.copy();initial_root[na]=pr[pa];initial_q[na]=pq[pa]
 hi,oldri=get_humanoid_retarget_indices_g1_27dof();ri=jnp.asarray([robot.links.names.index(G1_27_dof[int(i)]) for i in np.asarray(oldri)]);tree=create_conn_tree(robot,ri)
 plane=pk.collision.Heightmap(pose=jaxlie.SE3.identity(),size=jnp.array([.01,.01,1.]),height_data=jnp.zeros((100,100),dtype=jnp.float32));target=jnp.asarray(human[1:]);l,r=foot_detect(human[None],thres=(.05/fps)**2)
 receipt=json.loads((R/'pyroki_grab_temporal_20261030/report.json').read_text());weights=dict(receipt['weights']);weights['root_smoothness']=weights['foot_skating']=2*fps/20
 t=time.monotonic();outroot,outq=solve_retargeting(robot=robot,robot_coll=coll,target_keypoints=target,is_left_foot_contact=jnp.asarray(l[0]>0),is_right_foot_contact=jnp.asarray(r[0]>0),left_foot_keypoints=plane.project_points(target[:,10,:]),right_foot_keypoints=plane.project_points(target[:,11,:]),smpl_joint_retarget_indices=hi,g1_joint_retarget_indices=ri,smpl_mask=tree,heightmap=plane,weights=weights,pick_point=jnp.zeros(3),gmr_root=jnp.asarray(initial_root),gmr_joints=jnp.asarray(initial_q),anchor_root=jnp.asarray(anchor_root),anchor_joints=jnp.asarray(anchor_q),anchor_mask=jnp.asarray(mask))
 outroot=np.asarray(outroot.wxyz_xyz);outq=np.asarray(outq)
 if outq.shape!=(7,29) or not np.isfinite(outq).all() or not np.isfinite(outroot).all():raise ValueError('invalid output')
 def compare(roots,joints):
  aa=Rotation.from_quat(pr[pa][:,[1,2,3,0]]);bb=Rotation.from_quat(roots[na][:,[1,2,3,0]])
  return {'joint_rad':float(np.max(np.abs(pq[pa]-joints[na]))),'root_m':float(np.max(np.linalg.norm(pr[pa,4:]-roots[na,4:],axis=1))),'root_angle_rad':float(np.max((aa.inv()*bb).magnitude()))}
 out=R/'gmr29_anchor_limit_v2_witness_20261103';out.mkdir(exist_ok=False)
 with (out/'result.npz').open('xb') as f:np.savez_compressed(f,indices=idx,root_wxyz_xyz=outroot,joint_pos=outq,joint_names=np.asarray(target_names))
 result={'source_sha256':sha(source),'previous_window_sha256':sha(prior),'baseline_window_sha256':sha(prior_next),'helper_sha256':sha(h),'result_sha256':sha(out/'result.npz'),'overlap_frames':common.tolist(),'baseline':compare(br,bq),'anchored':compare(outroot,outq),'anchor_weight':10.0,'elapsed_s':time.monotonic()-t,'training_ready':False,'limitations':['One three-frame overlap only; anchor weights are project experiments, not author method.','Small overlap does not guarantee physical tracking, accuracy or long-clip continuity.','GRAB is not physically filtered.']}
 with (out/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
