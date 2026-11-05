"""Re-read all candidate files and bind independently reconstructed task targets."""
import hashlib,json,pathlib,sys
import numpy as np
import torch
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'pilot_v2';S=R/'stage3_pick29_20261103'
sys.path[:0]=[str(B/'src'),str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_supported_task_conditions import build,TASKS,PROMPTS
from stage3_pick29_core import BODY_JOINTS
from g1_fk import URDFFK
assert not (W/'candidate_audit.json').exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert json.loads((W/'status.json').read_text())['status']=='three_task_pilot_candidates_generated_postoptimization_pending'
q=np.zeros((1,29))
for side in ('left','right'):
 for part,value in [('hip_pitch',-.2),('knee',.42),('ankle_pitch',-.23),('elbow',.87),('shoulder_pitch',.35)]:q[0,BODY_JOINTS.index(side+'_'+part+'_joint')]=value
q[0,BODY_JOINTS.index('left_shoulder_roll_joint')]=.16;q[0,BODY_JOINTS.index('right_shoulder_roll_joint')]=-.16
urdf=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
initial,rot=URDFFK(urdf).forward(np.array([[0,0,.84]]),np.array([[0,0,0,1.]]),q,BODY_JOINTS,return_rotations=True)
rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.05,0)
all_rows=[];errors={}
for n,task in enumerate(TASKS):
 rows=[json.loads(x) for x in (W/task/'records.jsonl').read_text().splitlines()];assert len(rows)==2
 ids=list(range(100000+n*1000,100002+n*1000));assert [r['candidate_id'] for r in rows]==ids
 expected=build(task,ids,20261104,initial[0],rot[0]);local_errors=[]
 for k,r in enumerate(rows):
  p=pathlib.Path(r['path']);assert sha(p)==r['sha256']
  with np.load(p,allow_pickle=False) as z:
   meta=json.loads(z['metadata_utf8'].tobytes());feature=z['features'];point=z['canonical_points'];target=z['target_points'];mask=z['target_mask']
   assert feature.shape==(1,180,263) and point.shape==(1,180,22,3) and target.shape==(180,22,3)
   assert np.isfinite(feature).all() and np.isfinite(point).all() and meta['fps']==20 and meta['frames']==180
   assert meta['task']==task and meta['caption']==PROMPTS[task] and meta['scene_sha256']==expected['scene_sha256']
   assert meta['checkpoint_sha256']=='109fcfebf2a805831444ca8fb55d69674ef7cb19c61219e3e4ed18876d9527a9'
   assert meta['task_conditions_source_sha256']==sha(B/'src/stage3_supported_task_conditions.py')
   assert np.allclose(target,expected['world'][k]@rotation,atol=1e-7,rtol=0) and np.array_equal(mask[0],expected['mask'][k])
   for key in ['desired_object_positions','desired_wrist_quat_wxyz','planned_hand_closed']:assert np.allclose(z[key],expected[key][k],atol=1e-10,rtol=0)
   assert hashlib.sha256(feature[0].tobytes()).hexdigest()==r['feature_content_sha256']==meta['feature_content_sha256']
   noise=torch.randn((1,263,1,180),generator=torch.Generator(device='cuda:0').manual_seed(20261104+r['candidate_id']),device='cuda:0')
   assert hashlib.sha256(noise[0].cpu().numpy().tobytes()).hexdigest()==r['initial_noise_sha256']==meta['initial_noise_sha256']
   wrist_indices=[21] if task=='pour' else [20,21]
   local_errors.append({str(j):float(np.linalg.norm(point[0,:,j]-target[:,j],axis=-1).mean()) for j in wrist_indices})
  all_rows.append({'task':task,**r})
 errors[task]=local_errors
for key in ['candidate_id','feature_content_sha256','initial_noise_sha256']:assert len({r[key] for r in all_rows})==6
report={'status':'passed_candidate_file_noise_and_task_binding_audit','candidate_count':6,'accepted_reference_count':0,
 'formal_training_allowed':False,'body_dof_target':29,'fps':20,'frames':180,'source_condition_sha256':sha(B/'src/stage3_supported_task_conditions.py'),
 'records':all_rows,'raw_diffusion_mean_wrist_errors_m':errors,
 'scope':'raw three-task pilot candidates only; 29DoF postoptimization, hand/object scene calibration and final admission pending'}
(W/'candidate_audit.json').open('x').write(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'candidate_count':6,'accepted_reference_count':0}))
