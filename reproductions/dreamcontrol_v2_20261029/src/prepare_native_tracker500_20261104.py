"""Stage the full native trainer and a20Hz500-reference CSV adapter. CPU only."""
import ast,hashlib,json,os,shutil
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';W=D/'native_universal_20261104'
T=Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024/step34_20261029')
UP=T/'user_supermimic_vibed_source_20261103_v1';AD=T/'user_vibed_textop_adapter_20261103_v1';ENV=T/'user_vibed_native_env_20261103_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (W/'handoff_receipt.json').exists() and not (W/'preparation_receipt.json').exists()
source=W/'source';(source/'supermimic').mkdir(parents=True,exist_ok=False)
bindings={};deltas={}
for p in sorted((UP/'supermimic').glob('*.py')):
 chosen=AD/'supermimic'/p.name if p.name in ['env.py','robot.py'] else p
 target=source/'supermimic'/p.name;shutil.copyfile(chosen,target);bindings[str(chosen)]=sha(chosen)
 if p.name in ['env.py','robot.py']:deltas[p.name]='reuse previously validated TextOp asset/IsaacLab3 compatibility adapter; training optimizer code unchanged'
motion=source/'supermimic/motion.py';raw=motion.read_text();old='src_fps: float = 30.0';assert raw.count(old)==1
motion.write_text(raw.replace(old,'src_fps: float = 20.0'));deltas['motion.py']='only source-FPS constructor default30->20; all500 inputs keep their actual20Hz grid'
assert sha(source/'supermimic/train.py')==sha(UP/'supermimic/train.py')=='b94569605bd3ff96a49d4903bb791bb54bc4695a03f1914c8447c0ec7dd1e6bd'
os.symlink(str(AD/'data'),str(source/'data'),target_is_directory=True)
node=ast.parse((source/'supermimic/robot.py').read_text());names=ast.literal_eval(next(x.value for x in node.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='G1_JOINT_NAMES' for t in x.targets)))
assert len(names)==29 and len(set(names))==29
manifest=S/'reference_scale500_20261104/final_library_v1/reference_bank.json';m=json.loads(manifest.read_text());assert len(m['references'])==500
csv=W/'motion_csv20';csv.mkdir(exist_ok=False);rows=[]
for index,item in enumerate(m['references']):
 assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2
 assert sha(item['path'])==item['sha256']
 with np.load(item['path'],allow_pickle=False) as z:
  q=z['q'];p=z['root_pos'];rot=z['root_quat'];joint_names=z['joint_names'].tolist();times=z['sample_times']
  assert q.shape==(120,29) and np.allclose(times,np.arange(120)/20) and float(z['fps'])==20
  order=[joint_names.index(n) for n in names];matrix=np.c_[p,rot[:,[1,2,3,0]],q[:,order]]
  assert matrix.shape==(120,36) and np.isfinite(matrix).all()
  finger=z['hand_closed'].copy()
 target=csv/('pick_'+str(index).zfill(4)+'.csv');np.savetxt(target,matrix,delimiter=',',fmt='%.9g')
 check=np.loadtxt(target,delimiter=',');assert np.max(np.abs(check-matrix))<1e-7
 rows.append({'csv':str(target),'csv_sha256':sha(target),'source':item['path'],'source_sha256':item['sha256'],
  'source_fps':20,'source_frames':120,'body_joint_order':names,'hand_commands_retained_in_source':True,
  'object_position':item['task_object_position'],'note':'native body29 tracker input; no autonomous grasp or learned finger-control claim'})
(W/'native_motion_manifest.json').open('x').write(json.dumps({'source_manifest_sha256':sha(manifest),'records':rows},indent=2))
isaac=T/'user_vibed_isaaclab3_source_20261103_v1/source';extra=ENV/'isaaclab_extra_source_20261103_v1/source/source'
paths=[str(source),str(ENV/'newton_source_71eeba_20261103_v1/source'),str(ENV/'isaaclab_contrib_source_20261103_v1/source/source/isaaclab_contrib')]+[str(isaac/x) for x in ['isaaclab','isaaclab_newton','isaaclab_assets']]+[str(extra/x) for x in ['isaaclab_ov','isaaclab_physx','isaaclab_visualizers']]
spec={'kind':'warm reuse existing TextOp native SuperMimic environment, no installs','python':str(ENV/'venv/bin/python'),
 'native_upstream':'zita-ch/supermimic_vibed','native_commit':'0525a23a2dcfa61f8543f98b84c075fe66d9d286',
 'package_versions':json.loads((T/'user_vibed_controlled_runner_source_20261103_v5/environment_ledger.json').read_text())['packages'],
 'module_paths':paths,'full_native_train':str(source/'supermimic/train.py'),'full_native_train_sha256':sha(source/'supermimic/train.py'),
 'source_bindings':bindings,'adapter_changes':deltas,'input_csv_directory':str(csv),'native_source_fps':20,
 'policy_mode':'one universal reference-conditioned tracker; 10 history frames and10 future reference frames',
 'body_dof':29,'native_learned_actions':29,'hands_objects':'retained source metadata; native basic tracker does not yet model finger/object interaction',
 'gpu_uuid':'GPU-f3cc20c6-f613-73c1-b0f9-f54863f22602','gpu_per_job_bytes':12*1024**3,'rss_per_job_bytes':32*1024**3,
 'runtime_disk_floor':8*1024**3+356*1024**2,'PPO_process_unchanged':1436632,'formal_training_started':False,
 'cache_cap_bytes':1024**3,'output_cap_bytes':600*1024**2,'compile_threads':1,'batch_test':{'wm_batch':2048,'policy_batch':2048,'horizon':32,'AMP':'bf16','TF32':True}}
canonical=json.dumps(spec,sort_keys=True,separators=(',',':'));(W/'environment_spec.json').open('x').write(json.dumps({'spec_hash':hashlib.sha256(canonical.encode()).hexdigest()[:8],'spec':spec},indent=2))
receipt={'status':'native_full_trainer_and500_inputs_staged_not_training_ready','references':len(rows),'frames_at20Hz':60000,
 'native_internal_control_fps':50,'canonical_root_format':'CSVxyz/xyzw; source file WXYZ retained','native_learned_body_actions':29,
 'unchanged_training_core':['train.py','networks.py','mdp.py','buffer.py','kinematics.py','reset_sampling.py','rotations.py'],
 'full_native_train_sha256':sha(source/'supermimic/train.py'),'native_motion_manifest_sha256':sha(W/'native_motion_manifest.json'),
 'spec_hash':hashlib.sha256(canonical.encode()).hexdigest()[:8],'model_and_hand_scope_not_misrepresented':True}
(W/'preparation_receipt.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
