import ast,hashlib,json,pathlib,sys
import numpy as np
from scipy.spatial.transform import Rotation
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';W=R/'stage3_supported_references_20261104/pilot_v1';D=R/'stage2_prior29_20261103'
sys.path[:0]=[str(W.parent/'src'),str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_supported_task_conditions import build,TASKS,PROMPTS
from stage3_pick29_core import BODY_JOINTS
from g1_fk import URDFFK
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert not W.exists();W.mkdir(parents=True)
for name in ['stage3_supported_task_conditions.py','stage3_sample_supported29_batch.py']:ast.parse((W.parent/'src'/name).read_text())
cfg=json.loads((D/'config.json').read_text());manifest=pathlib.Path(cfg['manifest']);assert sha(manifest)==cfg['manifest_sha256']
rows=json.loads(manifest.read_text())['records'];rows=[r for r in rows if r['split']=='train']
urdf=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf')
assert sha(urdf)=='572b54991d8dccd6d3e091570788fb307bd0877b6a062426583f0a17fb56cc95'
q=np.zeros((1,29))
for side in ('left','right'):
 for part,value in [('hip_pitch',-.2),('knee',.42),('ankle_pitch',-.23),('elbow',.87),('shoulder_pitch',.35)]:q[0,BODY_JOINTS.index(side+'_'+part+'_joint')]=value
q[0,BODY_JOINTS.index('left_shoulder_roll_joint')]=.16;q[0,BODY_JOINTS.index('right_shoulder_roll_joint')]=-.16
points,rots=URDFFK(urdf).forward(np.array([[0,0,.84]]),np.array([[0,0,0,1.]]),q,BODY_JOINTS,return_rotations=True)
report={'status':'task_condition_plans_checked_not_generated','body_dof':29,'fps':20,'frames':180,'manifest_sha256':sha(manifest),'tasks':{},'formal_SuperMimic_training':False}
for n,task in enumerate(TASKS):
 ids=list(range(100000+n*1000,100002+n*1000));c=build(task,ids,20261104,points[0],rots[0])
 matches=[r['sample_id'] for r in rows if PROMPTS[task] in r.get('candidate_captions',[r['caption']])];assert matches,'prompt not found in current training captions'
 assert c['world'].shape==(2,180,22,3) and c['mask'].shape==(2,180,22)
 assert np.isfinite(c['world']).all() and np.allclose(np.linalg.norm(c['desired_wrist_quat_wxyz'],axis=-1),1)
 assert np.allclose(c['world'][:,0,[20,21]],points[:,[20,21]])
 obj=c['desired_object_positions'];initial=c['object_initial']
 if task=='pour':
  tilt=Rotation.from_quat(c['desired_wrist_quat_wxyz'][0,90,1,[1,2,3,0]])*Rotation.from_quat(c['desired_wrist_quat_wxyz'][0,70,1,[1,2,3,0]]).inv()
  assert abs(tilt.magnitude()-np.deg2rad(75))<1e-6 and np.allclose(obj[:,160],initial)
  assert c['planned_hand_closed'][:,90,1].all()
 elif task=='bimanual_box_lift':
  assert np.allclose(obj[:,70,2]-initial[:,2],.14) and np.allclose(obj[:,110,0]-initial[:,0],.1)
  assert np.allclose(obj[:,150,2],initial[:,2]) and c['planned_hand_closed'][:,70].all()
  assert np.allclose(c['world'][:,70,20,1]-c['world'][:,70,21,1],.43)
 else:
  assert np.allclose(obj[:,70,0]-initial[:,0],.1) and np.allclose(obj[:,144],initial)
  assert not c['planned_hand_closed'][:,90].any() and c['planned_hand_closed'][:,104].all()
 output=W/(task+'_planned_conditions.npz')
 with output.open('xb') as f:np.savez_compressed(f,world=c['world'],mask=c['mask'],desired_object_positions=obj,desired_wrist_quat_wxyz=c['desired_wrist_quat_wxyz'],planned_hand_closed=c['planned_hand_closed'])
 report['tasks'][task]={'candidate_ids':ids,'prompt':PROMPTS[task],'exact_training_caption_matches':matches,'scene':c['scene'],'scene_sha256':c['scene_sha256'],'plans':c['plans'],'planned_arrays':str(output),'planned_arrays_sha256':sha(output),'accepted_references':0}
report['source_sha256']={p.name:sha(p) for p in (W.parent/'src').glob('*.py')}
(W/'condition_preparation_report.json').open('x').write(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'task_ids':TASKS,'planned_candidates':6,'accepted_references':0}))
