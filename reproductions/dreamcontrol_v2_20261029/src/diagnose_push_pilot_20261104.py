import json,pathlib,sys,subprocess,shutil
import numpy as np
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';sys.path[:0]=[str(S/'src'),str(R/'training_disk_guard_20261101')]
from stage3_pick29_core import BODY_JOINTS
from g1_fk import URDFFK
fk=URDFFK(pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf'))
rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
for i in [102000,102001]:
 p=B/f'pilot_postopt_v4/candidate_{i}/report.json';r=json.loads(p.read_text())
 with np.load(r['output'],allow_pickle=False) as z,np.load(r['sample'],allow_pickle=False) as src:
  points=fk.forward(z['root_pos'],z['root_quat'][:,[1,2,3,0]],z['q'],BODY_JOINTS);goal=src['target_points']@rotation.T
  e=np.linalg.norm(points[:,20:22]-goal[:,20:22],axis=-1);t,side=np.unravel_index(e.argmax(),e.shape)
  print(json.dumps({'id':i,'max_error_time_s':float(t/20),'wrist':int(side),'wrist_actual':points[t,20+side].tolist(),'wrist_target':goal[t,20+side].tolist(),'root_at_worst':z['root_pos'][t].tolist(),'foot_positions_at_worst':points[t,10:12].tolist(),'checks':r['checks'],'sole_max':r['maximum_foot_sole_height_m']}))
for task in ['pour','bimanual_box_lift','box_push_pull']:
 j=json.loads((B/f'pilot_v2/{task}/report.json').read_text());print('SPEED',task,j['scene_sha256'])
 p=next((B/f'pilot_v2/{task}').glob('batch_*/report.json'));j=json.loads(p.read_text());print('SAMPLE_SECONDS',task,j['elapsed_s'])
print('PPO',json.loads((S/'runs/pick500_8912_3000_v1/status.json').read_text())['policy_updates'])
print('GPU',subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader,nounits'],text=True));print('DISK',shutil.disk_usage('/data').free)
