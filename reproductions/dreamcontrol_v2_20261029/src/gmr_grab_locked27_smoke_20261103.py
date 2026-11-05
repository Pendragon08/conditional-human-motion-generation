"""GRAB raw SMPL-X -> GMR 27DoF source-rate isolated technical smoke.

Original ZIP is read-only; this is not a physical or training acceptance.
"""
import contextlib,hashlib,io,json,shutil,sys,time,zipfile
from pathlib import Path
import mujoco,numpy as np,smplx,torch,trimesh
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
GMR=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR')
MODEL=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/remaining_20261021/SMPLX_lockedhead_20230207/extracted/models_lockedhead')
ROBOT=ROOT/'gmr_locked27_model_20261103/g1_gmr_locked27.xml';ROBOT_REPORT=ROOT/'gmr_locked27_model_20261103/report.json'
OUT=ROOT/'gmr_grab_locked27_smoke_20261103'
sys.path.insert(0,str(GMR));sys.path.insert(0,str(ROOT/'grab_full_human_work_v10_20261103'))
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.params import ROBOT_XML_DICT,IK_CONFIG_DICT
from general_motion_retargeting.utils.smpl import get_smplx_data
from grab_restricted_reader import read_grab_field
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def guard():
 if shutil.disk_usage(ROOT).free<8*1024**3+256*1024**2:raise RuntimeError('disk budget')
def main():
 torch.set_num_threads(1);guard()
 archive=ROOT/'data/GRAB/grab__s2.zip';member='s2/airplane_fly_1.npz';expected='d77eb7d13cfb3e3d0b81f08ccf9d8d79320fa6541352e6e65173c56d4b01b9b6';indices=np.arange(540,660,dtype=np.int64)
 model_report=json.loads(ROBOT_REPORT.read_text());robot_hash=sha(ROBOT)
 if sha(archive)!=expected or robot_hash!=model_report['model_sha256'] or model_report['nu']!=27:raise ValueError('source or GMR model changed')
 with zipfile.ZipFile(archive) as z:
  raw=z.read(member);member_hash=hashlib.sha256(raw).hexdigest();body=read_grab_field(z,member,'body');obj=read_grab_field(z,member,'object')
  with np.load(io.BytesIO(raw),allow_pickle=False) as f:fps=float(f['framerate'].item());frames=int(f['n_frames'].item());gender=str(f['gender'].item());pca=int(f['n_comps'].item())
 if fps!=120 or frames<=int(indices[-1]) or gender!='male' or pca!=24 or body['vtemp']!='tools/subject_meshes/male/s2.ply':raise ValueError('source metadata mismatch')
 template=ROOT/'grab_mesh_stage_20261030'/body['vtemp'];template_hash=sha(template);mesh=trimesh.load(str(template),process=False);vertices=np.asarray(mesh.vertices)
 height=float(np.ptp(vertices[:,2]))
 if vertices.shape!=(10475,3) or not np.isfinite(vertices).all() or not 1.2<=height<=2.2:raise ValueError('invalid neutral template height')
 names=('global_orient','body_pose','left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression','transl')
 tensors={k:torch.as_tensor(np.asarray(body['params'][k])[indices],dtype=torch.float32) for k in names}
 with contextlib.redirect_stdout(sys.stderr):
  model=smplx.create(str(MODEL),model_type='smplx',gender=gender,num_betas=10,num_pca_comps=pca,batch_size=len(indices),use_pca=True,ext='npz')
  if not np.array_equal(mesh.faces,model.faces):raise ValueError('template topology differs')
  model.v_template.copy_(torch.as_tensor(vertices,dtype=model.v_template.dtype))
  with torch.no_grad():human=model(**tensors,return_full_pose=True,return_verts=False)
  ROBOT_XML_DICT['g1_project_locked27']=ROBOT;IK_CONFIG_DICT['smplx']['g1_project_locked27']=IK_CONFIG_DICT['smplx']['unitree_g1']
  solver=GeneralMotionRetargeting(src_human='smplx',tgt_robot='g1_project_locked27',actual_human_height=height,verbose=False)
 joints=model_report['joint_names'];addresses=[]
 for name in joints:
  jid=mujoco.mj_name2id(solver.model,mujoco.mjtObj.mjOBJ_JOINT,name)
  if jid<0:raise ValueError('missing joint')
  addresses.append(int(solver.model.jnt_qposadr[jid]))
 result=[];started=time.monotonic()
 for i in range(len(indices)):
  if i%16==0:guard()
  q=solver.retarget(get_smplx_data({},model,human,i));result.append(np.r_[q[3:7],q[:3],q[addresses]])
  if (i+1)%30==0:print('frames',i+1,flush=True)
 result=np.asarray(result,dtype=np.float64)
 if result.shape!=(len(indices),34) or not np.isfinite(result).all() or not np.allclose(np.linalg.norm(result[:,:4],axis=1),1,atol=1e-5,rtol=0):raise ValueError('invalid output')
 if sha(ROBOT)!=robot_hash or sha(template)!=template_hash or sha(archive)!=expected:raise RuntimeError('input changed')
 OUT.mkdir(exist_ok=False);target=OUT/'gmr_source_rate.npz'
 with target.open('xb') as f:np.savez_compressed(f,source_frames=indices,declared_fps=np.array(fps),root_wxyz_xyz=result[:,:7],joint_pos=result[:,7:],joint_names=np.asarray(joints),human_joints=human.joints[:,:22].detach().cpu().numpy(),object_transl=np.asarray(obj['params']['transl'])[indices],object_orient_rotvec=np.asarray(obj['params']['global_orient'])[indices])
 report={'member':member,'source_member_sha256':member_hash,'archive_sha256':expected,'template_sha256':template_hash,'model_sha256':sha(MODEL/'smplx/SMPLX_MALE.npz'),'gmr_robot_sha256':robot_hash,'gmr_config_sha256':sha(IK_CONFIG_DICT['smplx']['unitree_g1']),'output_sha256':sha(target),'frames':len(indices),'source_fps_declared':fps,'human_template_height_m':height,'elapsed_s':time.monotonic()-started,'training_ready':False,'limitations':['GMR sequential IK, project-local locked-27 model and template-bounds height approximation.','Source FPS metadata, feet contact, collision, text, actual clock and physics are not validated.','GMR model differs from DreamControl-v2 author method; no 20Hz output yet.']}
 with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,allow_nan=False)
 print(json.dumps({'frames':len(indices),'height_m':height,'elapsed_s':report['elapsed_s'],'training_ready':False}),flush=True)
if __name__=='__main__':main()
