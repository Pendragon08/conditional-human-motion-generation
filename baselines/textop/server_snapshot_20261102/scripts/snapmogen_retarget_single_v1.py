"""One real SnapMoGen source through pinned ScaleRetarget; no PPO or simulation rollout."""
import argparse,fcntl,hashlib,json,os,pathlib,sys
if sys.flags.optimize or not sys.flags.isolated:raise RuntimeError('isolated interpreter required')
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--trusted-request-sha256',required=True);a=parser.parse_args()
p=pathlib.Path(a.request)
if os.getuid()!=1013 or p.resolve()!=p or p.stat().st_uid!=1013:raise ValueError('request identity')
raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=a.trusted_request_sha256:raise ValueError('request hash')
j=json.loads(raw);role=j['role'];run=j['run'];out=ROOT/'step34_20261029'/run
if role not in ('author','document') or run!='user_snapmogen_retarget_single_'+role+'_v1' or out.resolve()!=out or out.stat().st_uid!=1013:raise ValueError('scope')
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'scripts'))
from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
lockpath=ROOT/'receipts/full_library_capacity.lock';s=lockpath.stat();parent=os.getppid()
if not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(),parent,os.major(s.st_dev),os.minor(s.st_dev),s.st_ino):raise ValueError('supervisor lock not held')
env=ROOT/'step34_20261029/user_snapmogen_python311_runtime_v1/env_v1';site=env/'site';source=env/'source/ScaleRetarget';sample=ROOT/'step34_20261029/user_snapmogen_single_source_v1/m_st3_00000.bvh'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def verify_sources():
 manifest=env/'meta/source_manifest.json'
 if sha(manifest)!=j['source_manifest_sha256']:raise ValueError('source manifest')
 m=json.loads(manifest.read_text())
 for x in m['items']:
  f=env/'source'/x['path']
  if f.is_symlink() or f.stat().st_size!=x['bytes'] or sha(f)!=x['sha256']:raise ValueError('source changed: '+x['path'])
 return sha(manifest)
if sample.resolve()!=sample or sample.is_symlink() or sample.stat().st_uid!=1013 or sample.stat().st_size!=307721 or sha(sample)!=j['source_sha256']:raise ValueError('input binding')
if sha(ROOT/'scripts/snapmogen_mink_compat_v1.py')!=j['compat_sha256']:raise ValueError('adapter binding')
source_before=verify_sources();sys.path[:0]=[str(source),str(site)]
import numpy as np
import torch
import mujoco
import mink
from omegaconf import OmegaConf
from scaleretarget.loader.snapmogen_loader import SnapMogenLoader
from scaleretarget.retargeter.gmr_retargeter import GmrRetargeter
from scaleretarget.formatter.kinematic_formatter import KinematicFormatter
from snapmogen_mink_compat_v1 import MinkLimitCompatibility
if torch.version.cuda is not None or torch.__version__!='2.5.1+cpu' or mujoco.__version__!='3.3.7':raise ValueError('environment differs')
for module in (np,torch,mujoco,mink):
 if site not in pathlib.Path(module.__file__).resolve().parents:raise ValueError('package origin')
torch.set_num_threads(4);torch.manual_seed(20261031);np.random.seed(20261031)
asset=source/'assets/unitree_g1/g1_mocap_29dof.xml'
if sha(asset)!=j['asset_sha256']:raise ValueError('asset hash')
robot=dict(robot_type='unitree_g1',robot_xml_path=str(asset),robot_root_height=.79,robot_total_height=1.32)
config=OmegaConf.create(dict(data_format='.bvh',target_fps=30,use_optimized_shape=True,optim_iterations=3000,
 optim_joint_matches=str(source/'config/correspondence/optim_shape/snapmogen_to_unitree_g1.yaml'),robot=robot))
log=out/'progress.jsonl'
def event(v):
 with log.open('a',encoding='utf-8') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
event(dict(phase='SHAPE_FITTING',steps=0))
count=[0];oldstep=torch.optim.Adam.step
def measured_step(self,*args,**kwargs):
 result=oldstep(self,*args,**kwargs);count[0]+=1
 if count[0]%100==0:event(dict(phase='SHAPE_FITTING',steps=count[0]))
 return result
torch.optim.Adam.step=measured_step
try:
 loader=SnapMogenLoader(config);loader.load(str(sample))
finally:torch.optim.Adam.step=oldstep
if count[0]!=3000 or not np.isfinite(loader.scale).all() or not (loader.scale>0).all():raise ValueError('shape fitting incomplete')
frames,extras=loader._load_sample(str(sample))
if len(frames)!=406 or not (29.99<float(extras['fps'])<30.01):raise ValueError('source timing')
event(dict(phase='RETARGETING',frames=0,total=len(frames),shape_steps=count[0]))
rcfg=OmegaConf.create(dict(robot=robot,viewer=dict(enable_viewer=False,record_video=False,camera_follow=True,rate_limit=False),
 ik_config=str(source/'config/correspondence/gmr/snapmogen_to_unitree_g1.yaml'),solver='daqp',damping=.5,use_velocity_limit=False))
retargeter=GmrRetargeter(rcfg);retargeter.update(extras)
compat=MinkLimitCompatibility(mink);compat.install();orig=retargeter.retarget_frame;done=[0]
def measured_frame(*args,**kwargs):
 value=orig(*args,**kwargs);done[0]+=1
 if done[0]%10==0:event(dict(phase='RETARGETING',frames=done[0],total=len(frames),shape_steps=count[0]))
 return value
retargeter.retarget_frame=measured_frame
try:qpos=retargeter.retarget(frames)
finally:compat.restore()
if done[0]!=406 or qpos.shape!=(406,36) or not np.isfinite(qpos).all() or compat.corrected_calls<406:raise ValueError('retarget incomplete')
model=retargeter.model;names=[]
for index in range(model.njnt):
 if int(model.jnt_type[index])==int(mujoco.mjtJoint.mjJNT_FREE):continue
 if int(model.jnt_type[index])!=int(mujoco.mjtJoint.mjJNT_HINGE):raise ValueError('non-hinge joint')
 names.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,index))
if len(names)!=29 or len(set(names))!=29:raise ValueError('joint identity')
formatter=KinematicFormatter(OmegaConf.create(dict(robot=robot,kinematic_model_device='cpu',height_adjust=True,root_offset=False,quat_order='xyzw')))
formatted=formatter.format(qpos.copy(),extras)
rootpos=np.asarray(formatted['root_pos'],dtype=np.float32);rootrot=np.asarray(formatted['root_rot'],dtype=np.float32);dof=np.asarray(formatted['dof_pos'],dtype=np.float32)
output=np.concatenate((rootpos,rootrot[:,[3,0,1,2]],dof),axis=1)
if output.shape!=(406,36) or not np.isfinite(output).all() or np.max(np.abs(np.linalg.norm(output[:,3:7],axis=1)-1))>1e-4:raise ValueError('output arrays')
if sha(sample)!=j['source_sha256'] or verify_sources()!=source_before:raise ValueError('input or source changed')
target=out/'reference.private.npz';partial=out/'reference.private.npz.partial'
with partial.open('xb') as f:np.savez_compressed(f,qpos_29=output,joint_names=np.asarray(names),fps=np.asarray(float(extras['fps'])),source_start_frame=np.asarray(1));f.flush();os.fsync(f.fileno())
if target.exists():raise ValueError('output exists')
os.rename(partial,target)
result=dict(status='SNAPMOGEN_ONE_SOURCE_RETARGETED_AWAITING_INDEPENDENT_AUDIT',role=role,source_sha256=j['source_sha256'],source_total_frames=410,
 source_frame_range=[1,407],source_fps=float(extras['fps']),retargeted_frames=len(output),first_test_clip='m_st3_00000#0#282',
 first_test_clip_retained_source_range=[1,282],first_test_clip_missing_leading_frames=1,shape_optimizer_steps=count[0],
 shape_scale_min=float(loader.scale.min()),shape_scale_max=float(loader.scale.max()),mink_limit_adapter_calls=compat.corrected_calls,
 reference_bytes=target.stat().st_size,reference_sha256=sha(target),qpos_array_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
 joint_names=names,asset_sha256=j['asset_sha256'],source_manifest_sha256=source_before,policy_loaded=False,physics_rollout_steps=0)
with (out/'receipt.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
event(dict(phase='FINISHED',frames=406,total=406,shape_steps=count[0]));print(json.dumps(result))
