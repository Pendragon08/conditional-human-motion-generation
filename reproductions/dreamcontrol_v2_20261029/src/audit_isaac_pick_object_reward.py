"""AST configuration and scalar audit of old Pick reset and height rewards."""
import ast
import hashlib
import json
import sys
from pathlib import Path
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
BASE=TRAIN/'source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_interactive_base.py';PICK=BASE.with_name('motion_tracking_pick_env.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=Path(sys.argv[1]);r=json.loads(p.read_text());assert sha(BASE)==r['base_sha256'] and sha(PICK)==r['pick_sha256']
t=ast.parse(PICK.read_text());flags={n.targets[0].id:ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('TRACKING','TASK_SPARSE','TASK_DENSE')}
def assignment(class_name,var):
 c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==class_name)
 return next(n for n in ast.walk(c) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==var).value
reset={k.arg:ast.literal_eval(k.value) for k in assignment('EventCfg','reset_object').keywords if k.arg in ('params','mode')}
reward={k.arg:ast.literal_eval(k.value) for k in assignment('G1Rewards','object_above_the_ground').keywords if k.arg in ('params','weight')}
assert flags['TASK_SPARSE'] is True and reset['params']['height']==1. and reset['mode']=='reset'
assert reward['params']=={'height_thres':.95,'fall_thres':.9} and reward['weight']==.5
base=ast.parse(BASE.read_text());f=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='object_above_threshold')
comparisons=[ast.unparse(x) for x in ast.walk(f) if isinstance(x,ast.Compare)]
assert 'root_pos[:, 2] > height_thres' in comparisons and 'root_pos[:, 2] < height_thres' in comparisons
assert r['outputs']['true']['reward'][2:]==[0.,1.,1.] and r['reset_local_heights']==[1.,.25]
out={'status':'passed','report_sha256':sha(p),'base_sha256':sha(BASE),'pick_sha256':sha(PICK),'config_flags':flags,'reset':reset,'reward':reward,'threshold_comparisons':comparisons,'script_sha256':sha(Path(__file__)),'limitations':'Static AST and witness consistency only; no dynamic manager, PhysX, actual environment origins or closed-state trajectories checked.'}
with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
