"""Retarget the complete authenticated SnapMoGen test sources, preserving failures."""
import argparse,hashlib,json,os,pathlib,sys,time,traceback
if sys.flags.optimize or not sys.flags.isolated:raise RuntimeError('isolated interpreter required')
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');RUNS={'user_snapmogen_full_retarget_author_v1','user_snapmogen_full_retarget_document_v1','user_snapmogen_full_retarget_v1'}
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args();request=pathlib.Path(a.request)
if os.getuid()!=1013 or request.resolve()!=request or sha(request)!=a.trusted_request_sha256:raise ValueError('request identity')
j=json.loads(request.read_text());RUN=j['run'];out=ROOT/'step34_20261029'/RUN
if RUN not in RUNS or request!=ROOT/'receipts'/(RUN+'_request.json') or j['authorized'] is not True or out.resolve()!=out or out.stat().st_uid!=1013:raise ValueError('scope')
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'scripts'))
from augmentation5368_audit_binding_v1 import lock_owned_by_supervisor
lk=(ROOT/'receipts/full_library_capacity.lock').stat()
if not lock_owned_by_supervisor(pathlib.Path('/proc/locks').read_text().splitlines(),os.getppid(),os.major(lk.st_dev),os.minor(lk.st_dev),lk.st_ino):raise ValueError('supervisor lock')
for name,digest in j['bound_sources'].items():
 q=ROOT/'scripts'/name
 if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('source code changed '+name)
env=ROOT/'step34_20261029/user_snapmogen_python311_runtime_v1/env_v1';site=env/'site';source=env/'source/ScaleRetarget';inputs=ROOT/'step34_20261029/user_snapmogen_full_sources_v1'
def verify_sources():
 manifest=env/'meta/source_manifest.json'
 if sha(manifest)!=j['source_manifest_sha256']:raise ValueError('source manifest changed')
 for x in json.loads(manifest.read_text())['items']:
  f=env/'source'/x['path']
  if f.is_symlink() or f.stat().st_size!=x['bytes'] or sha(f)!=x['sha256']:raise ValueError('source changed '+x['path'])
 return sha(manifest)
source_before=verify_sources();sys.path[:0]=[str(source),str(site)]
import numpy as np
import torch,mujoco,mink
from omegaconf import OmegaConf
from scaleretarget.loader.snapmogen_loader import SnapMogenLoader
from scaleretarget.retargeter.gmr_retargeter import GmrRetargeter
from scaleretarget.formatter.kinematic_formatter import KinematicFormatter
from snapmogen_mink_compat_v1 import MinkLimitCompatibility
if torch.version.cuda is not None or torch.__version__!='2.5.1+cpu' or mujoco.__version__!='3.3.7' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU environment')
for module in (np,torch,mujoco,mink):
 if site not in pathlib.Path(module.__file__).resolve().parents:raise ValueError('package origin')
torch.set_num_threads(4);asset=source/'assets/unitree_g1/g1_mocap_29dof.xml'
if sha(asset)!=j['asset_sha256']:raise ValueError('G1 asset')
robot=dict(robot_type='unitree_g1',robot_xml_path=str(asset),robot_root_height=.79,robot_total_height=1.32)
config=OmegaConf.create(dict(data_format='.bvh',target_fps=30,use_optimized_shape=True,optim_iterations=3000,optim_joint_matches=str(source/'config/correspondence/optim_shape/snapmogen_to_unitree_g1.yaml'),robot=robot))
rcfg=OmegaConf.create(dict(robot=robot,viewer=dict(enable_viewer=False,record_video=False,camera_follow=True,rate_limit=False),ik_config=str(source/'config/correspondence/gmr/snapmogen_to_unitree_g1.yaml'),solver='daqp',damping=.5,use_velocity_limit=False))
formatter=KinematicFormatter(OmegaConf.create(dict(robot=robot,kinematic_model_device='cpu',height_adjust=True,root_offset=False,quat_order='xyzw')))
if len(j['sources'])!=(1120 if RUN=='user_snapmogen_full_retarget_v1' else 2) or sha(inputs/'test_ids.txt')!=j['split_sha256']:raise ValueError('input count or split')
def event(v):
 with (out/'journal.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
started=time.monotonic();manifest=[];failed=[]
for index,item in enumerate(j['sources']):
 name=item['name'];sample=inputs/name
 if pathlib.PurePosixPath(name).name!=name or sample.resolve()!=sample or sample.stat().st_uid!=1013 or sample.stat().st_size!=item['bytes'] or sha(sample)!=item['sha256']:raise ValueError('input differs '+name)
 if name in j['reuse_sources']:
  prior=pathlib.Path(j['reuse_sources'][name]['path'])
  if prior.resolve()!=prior or prior.stat().st_uid!=1013 or sha(prior)!=j['reuse_sources'][name]['sha256']:raise ValueError('reused source binding')
  manifest.append(dict(index=index,name=name,status='REUSED_AUDITED',reference=str(prior),reference_sha256=sha(prior),frames=item['frames']-4,source_sha256=item['sha256']))
  event(dict(index=index,name=name,status='REUSED_AUDITED'));continue
 event(dict(index=index,name=name,status='STARTED',elapsed_seconds=time.monotonic()-started))
 try:
  seed=20261031+index;torch.manual_seed(seed);np.random.seed(seed)
  count=[0];oldstep=torch.optim.Adam.step
  def measured(self,*args,**kwargs):
   result=oldstep(self,*args,**kwargs);count[0]+=1;return result
  torch.optim.Adam.step=measured
  try:loader=SnapMogenLoader(config);loader.load(str(sample))
  finally:torch.optim.Adam.step=oldstep
  if count[0]!=3000 or not np.isfinite(loader.scale).all() or not (loader.scale>0).all():raise ValueError('shape fitting')
  frames,extras=loader._load_sample(str(sample));fps=float(extras['fps'])
  if len(frames)!=item['frames']-4 or abs(fps-1/item['frame_time'])>1e-5:raise ValueError('source timing')
  retargeter=GmrRetargeter(rcfg);retargeter.update(extras);compat=MinkLimitCompatibility(mink);compat.install()
  try:qpos=retargeter.retarget(frames)
  finally:compat.restore()
  if qpos.shape!=(len(frames),36) or not np.isfinite(qpos).all() or compat.corrected_calls<len(frames):raise ValueError('IK output')
  model=retargeter.model;names=[]
  for k in range(model.njnt):
   if int(model.jnt_type[k])==int(mujoco.mjtJoint.mjJNT_FREE):continue
   if int(model.jnt_type[k])!=int(mujoco.mjtJoint.mjJNT_HINGE):raise ValueError('joint type')
   names.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,k))
  if len(names)!=29 or len(set(names))!=29:raise ValueError('joints')
  formatted=formatter.format(qpos.copy(),extras);rootpos=np.asarray(formatted['root_pos'],dtype=np.float32);rootrot=np.asarray(formatted['root_rot'],dtype=np.float32);dof=np.asarray(formatted['dof_pos'],dtype=np.float32)
  result=np.concatenate((rootpos,rootrot[:,[3,0,1,2]],dof),axis=1)
  if result.shape!=(len(frames),36) or not np.isfinite(result).all() or np.max(np.abs(np.linalg.norm(result[:,3:7],axis=1)-1))>1e-4 or sha(sample)!=item['sha256']:raise ValueError('output or source')
  target=out/(name+'.npz');part=out/(name+'.npz.partial')
  with part.open('xb') as f:np.savez_compressed(f,qpos_29=result,joint_names=np.asarray(names),fps=np.asarray(fps),source_start_frame=np.asarray(1));f.flush();os.fsync(f.fileno())
  if target.exists():raise ValueError('output exists')
  os.rename(part,target)
  record=dict(index=index,name=name,status='RETARGETED',reference=str(target),reference_sha256=sha(target),qpos_array_sha256=hashlib.sha256(result.tobytes()).hexdigest(),frames=len(frames),source_sha256=item['sha256'],shape_optimizer_steps=count[0],ik_calls=compat.corrected_calls,seed=seed)
  manifest.append(record);event(dict(index=index,name=name,status='RETARGETED',elapsed_seconds=time.monotonic()-started))
 except Exception as e:
  failed.append(dict(index=index,name=name,error=repr(e),traceback=traceback.format_exc()));event(dict(index=index,name=name,status='FAILED',error=repr(e)))
if verify_sources()!=source_before or sha(inputs/'test_ids.txt')!=j['split_sha256']:raise ValueError('sources changed after processing')
with (out/'manifest.json').open('x',encoding='utf-8') as f:json.dump(dict(sources=manifest,failures=failed),f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
rec=dict(status='SNAPMOGEN_FULL_RETARGET_FINISHED_AWAITING_AUDIT',total_sources=len(j['sources']),retargeted_or_reused=len(manifest),failed_sources=len(failed),manifest_sha256=sha(out/'manifest.json'),output_writers_closed=True,full_test_set_evaluated=False,policy_loaded=False,physics_rollout_steps=0,elapsed_seconds=time.monotonic()-started)
with (out/'receipt.json').open('x',encoding='utf-8') as f:json.dump(rec,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
print(json.dumps(rec))
