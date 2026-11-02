"""One audited SnapMoGen pilot clip through the official TextOp MuJoCo controller."""
import argparse,hashlib,json,os,pathlib,sys,time,traceback
if sys.flags.optimize or not sys.flags.isolated:raise RuntimeError('isolated interpreter required')
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');RUN='user_snapmogen_pilot_mujoco_v1'
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args()
request=pathlib.Path(a.request)
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if os.getuid()!=1013 or request.resolve()!=request or request.parent!=ROOT/'receipts' or sha(request)!=a.trusted_request_sha256:raise ValueError('request identity')
job=json.loads(request.read_text());out=ROOT/'step34_20261029'/RUN
if job['run']!=RUN or not job['authorized'] or out.resolve()!=out or out.stat().st_uid!=1013:raise ValueError('scope')
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'scripts'))
from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
st=(ROOT/'receipts/full_library_capacity.lock').stat()
if not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(),os.getppid(),os.major(st.st_dev),os.minor(st.st_dev),st.st_ino):raise ValueError('supervisor lock')
for name,digest in job['bound_sources'].items():
 q=ROOT/name
 if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('source changed: '+name)
envsite=ROOT/'step34_20261029/mixed5000_ort_overlay_v1/site';sys.path.insert(0,str(envsite))
import numpy as np
import scipy
from scipy.spatial.transform import Rotation,Slerp
import torch,mujoco,onnxruntime as ort
from run_mixed5000_official_mujoco_generated_v1 import official_functions
from augmentation2526_mixed5000_heldout_inputs_v1 import accept
from textop_paper_tracking_metrics_v2 import measure
torch.set_num_threads(4)
if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized() or ort.__version__!='1.20.1':raise ValueError('CPU-only environment')
accepted=accept(job['input_binding'])
if accepted['actor_sha256']!='f2ee798dd0ae3a9b7878c720742d444aa09747ea39829ed96a173f7e2de829c7':raise ValueError('policy binding')
policy=ROOT/'step34_20261029/user_mixed5000_onnx_export_v1/policy.onnx'
if sha(policy)!=job['onnx_sha256']:raise ValueError('ONNX binding')
asset=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR/assets/unitree_g1/g1_mocap_29dof.xml')
if sha(asset)!=job['asset_sha256']:raise ValueError('G1 asset')
for name,digest in job['mesh_hashes'].items():
 q=pathlib.Path(name)
 if q.resolve()!=q or asset.parent not in q.parents or sha(q)!=digest:raise ValueError('G1 mesh')
source=ROOT/'step34_20261029/user_snapmogen_retarget_single_author_v1/reference.private.npz'
if source.resolve()!=source or sha(source)!=job['retarget_reference_sha256']:raise ValueError('audited reference bytes')
with np.load(source,allow_pickle=False) as z:
 q=np.asarray(z['qpos_29'],dtype=np.float32);names=z['joint_names'].tolist();fps=float(z['fps']);start=int(z['source_start_frame'])
if q.shape!=(406,36) or start!=1 or hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()!=job['retarget_qpos_sha256'] or not 29.99<fps<30.01:raise ValueError('audited reference arrays')
original=q[:281];duration=(len(original)-1)/fps;times=np.arange(int(np.floor(duration*50))+1,dtype=np.float64)/50.
if len(times)<2 or times[-1]>duration+1e-12:raise ValueError('target time coverage')
source_times=np.arange(len(original),dtype=np.float64)/fps
resampled=np.empty((len(times),36),dtype=np.float64)
for col in list(range(3))+list(range(7,36)):resampled[:,col]=np.interp(times,source_times,original[:,col])
rot=Rotation.from_quat(original[:,[4,5,6,3]]);resampled[:,3:7]=Slerp(source_times,rot)(times).as_quat()[:,[3,0,1,2]]
if not np.isfinite(resampled).all() or np.max(np.abs(np.linalg.norm(resampled[:,3:7],axis=1)-1))>1e-6:raise ValueError('resampled pose')
official=official_functions(np,torch);model=mujoco.MjModel.from_xml_path(str(asset));model.opt.timestep=.002
model_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(1,model.njnt)]
if (model.nq,model.nv,model.nu)!=(36,35,29) or names!=model_names or names!=official['mujoco_joint_names']:raise ValueError('joint order')
fkrec=pathlib.Path('/home/ouyanghao/textop-user100k-evaluation-20261029/user_ldm100k_generated_fk_v2/receipt.json')
if sha(fkrec)!=job['input_binding']['fk_hashes']['receipt']:raise ValueError('body mapping receipt')
fk=json.loads(fkrec.read_text());body_names=fk['body_names'];ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in body_names]
if len(ids)!=30 or len(set(ids))!=30 or any(x<0 for x in ids) or body_names[0]!='pelvis' or fk['joint_names']!=official['isaaclab_joint_names']:raise ValueError('body mapping')
indices=[names.index(n) for n in official['isaaclab_joint_names']]
if not np.array_equal(indices,official['mujoco_to_isaaclab_reindex']):raise ValueError('joint reorder')
joint_pos=np.asarray(resampled[:,7:][:,indices],dtype=np.float32);joint_vel=np.gradient(joint_pos,.02,axis=0).astype(np.float32)
fkdata=mujoco.MjData(model);bodypos=[];bodyquat=[]
for frame in resampled:
 fkdata.qpos[:]=frame;mujoco.mj_forward(model,fkdata);bodypos.append(fkdata.xpos[ids].copy());bodyquat.append(fkdata.xquat[ids].copy())
bodypos=np.asarray(bodypos,dtype=np.float32);bodyquat=np.asarray(bodyquat,dtype=np.float32)
if not(np.isfinite(joint_pos).all() and np.isfinite(joint_vel).all() and np.isfinite(bodypos).all() and np.isfinite(bodyquat).all()):raise ValueError('reference finite')
ref=out/'pilot_reference.private.npz'
with ref.open('xb') as f:np.savez_compressed(f,fps=np.array([50]),joint_pos=joint_pos,joint_vel=joint_vel,body_pos_w=bodypos,body_quat_w=bodyquat);f.flush();os.fsync(f.fileno())
loader=official['MotionLoader'](str(ref));opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
session=ort.InferenceSession(str(policy),sess_options=opts,providers=['CPUExecutionProvider'])
if session.get_providers()!=['CPUExecutionProvider'] or loader.T!=len(times):raise ValueError('policy provider or reference')
input_name=session.get_inputs()[0].name;data=mujoco.MjData(model);action=np.zeros(29,dtype=np.float32);target=official['default_angles'].copy()
data.qpos[7:]=loader.joint_pos[0][official['isaaclab_to_mujoco_reindex']];data.qpos[:3]=loader.body_pos[0][0];data.qpos[3:7]=loader.body_ori[0][0];mujoco.mj_step(model,data)
poses=[];executed=[];reference=[]
for t in range(loader.T):
 for _ in range(10):
  tau=official['pd_control'](target,data.qpos[7:],official['kps'],np.zeros(29),data.qvel[6:],official['kds']);data.ctrl[:]=tau;mujoco.mj_step(model,data)
 if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or not np.isfinite(data.xpos[ids]).all():break
 poses.append(data.qpos.copy());executed.append(data.xpos[ids].copy());reference.append(loader.body_pos[t].copy())
 obs=official['compute_observation'](data,loader,t,action)
 if obs.shape!=(431,) or not np.isfinite(obs).all():raise ValueError('observation')
 action=session.run(None,{input_name:np.asarray(obs,dtype=np.float32).reshape(1,-1)})[0].reshape(-1)
 if action.shape!=(29,) or not np.isfinite(action).all():raise ValueError('policy action')
 target=action[official['isaaclab_to_mujoco_reindex']]*official['action_scale']+official['default_angles']
if len(poses)<2:raise ValueError('too few finite simulation frames')
metric=measure(np.asarray(executed),np.asarray(reference),loader.T)
trace=out/'pilot_trace.private.npz'
with trace.open('xb') as f:np.savez_compressed(f,qpos=np.asarray(poses,dtype=np.float32),executed_body_pos=np.asarray(executed,dtype=np.float32),reference_body_pos=np.asarray(reference,dtype=np.float32));f.flush();os.fsync(f.fileno())
if sha(source)!=job['retarget_reference_sha256'] or sha(policy)!=job['onnx_sha256']:raise ValueError('input changed')
result=dict(status='SNAPMOGEN_PILOT_MUJOCO_EVALUATED_AWAITING_AUDIT',run=RUN,output_writers_closed=True,optimizer_updates=0,clip='m_st3_00000#0#282',source_fps=fps,
 retained_source_frames=281,missing_leading_source_frames=1,source_duration_seconds=duration,reference_fps=50,reference_frames=loader.T,
 reference_duration_seconds=float(times[-1]),unused_source_tail_seconds=float(duration-times[-1]),
 reference_sha256=sha(ref),trace_sha256=sha(trace),metrics=metric,actor_sha256=accepted['actor_sha256'],onnx_sha256=job['onnx_sha256'],
 physics_timestep_seconds=.002,control_hz=50,official_controller_used=True,reference_fk='MuJoCo GMR asset',
 full_test_set_evaluated=False,full_original_clip_evaluated=False,physical_hardware_used=False)
with (out/'receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
