"""Run the old Pick reward function and actual math source on mock CPU states."""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
import torch

TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
PICK=TRAIN/'source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_pick_env.py'
MATH=TRAIN/'source/isaaclab/isaaclab/utils/math.py';ROT=TRAIN/'isaac_utils/isaac_utils/rotations.py';MATHS=TRAIN/'isaac_utils/isaac_utils/maths.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1)
# Only the logging dependency is stubbed; all quaternion/math routines are real.
omni=types.ModuleType('omni');omni.__path__=[];log=types.ModuleType('omni.log');omni.log=log;sys.modules['omni']=omni;sys.modules['omni.log']=log
spec=importlib.util.spec_from_file_location('isaac_math_isolated',str(MATH));math_utils=importlib.util.module_from_spec(spec);sys.modules[spec.name]=math_utils;spec.loader.exec_module(math_utils)
sys.path.insert(0,str(TRAIN/'isaac_utils'));from isaac_utils.rotations import slerp
tree=ast.parse(PICK.read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='target_orientation_error')
source=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),func],type_ignores=[]);ast.fix_missing_locations(source)
class Entity:
    def __init__(self,name='robot',body_ids=None):self.name=name;self.body_ids=[0] if body_ids is None else body_ids
ns={'torch':torch,'math_utils':math_utils,'slerp':slerp,'SceneEntityCfg':Entity}
exec(compile(source,str(PICK),'exec'),ns);fn=ns['target_orientation_error']
class Obj:pass
class Scene:
    def __init__(self,n):
        self.env_origins=torch.zeros((n,3));self.robot=Obj();self.robot.data=Obj();self.robot.data.body_state_w=torch.zeros((n,1,13));self.robot.data.body_quat_w=torch.tensor([1.,0.,0.,0.]).repeat(n,1,1)
    def __getitem__(self,key):assert key=='robot';return self.robot
class Library:
    def __init__(self,pos,closed):self.pos=pos;self.closed=closed
    def get_motion_state(self,ids,times):return {'grab_pos':self.pos,'offsets':torch.zeros_like(self.pos),'is_closed':self.closed}
targets=torch.tensor([[1.,.25,.5],[0.,0.,0.],[0.,0.,1.],[1e-8,0.,1.]],dtype=torch.float32)
out={}
for closed in (False,True):
    env=Obj();env.scene=Scene(len(targets));env.device='cpu';env.episode_length_buf=torch.zeros(len(targets));env.step_dt=.02;env.start_motion_times=torch.full((len(targets),),2.);env.motion_ids=torch.arange(len(targets));env.motion_lib=Library(targets,torch.full((len(targets),),closed,dtype=torch.bool))
    v=fn(env,Entity());out['closed_'+str(closed).lower()]=[float(x) if torch.isfinite(x) else None for x in v]
    out['finite_'+str(closed).lower()]=torch.isfinite(v).tolist()
assert out['finite_false'][0] and not out['finite_false'][1] and not out['finite_false'][2]
result={'target_vectors_m':targets.tolist(),'results':out,'pick_sha256':sha(PICK),'isaac_math_sha256':sha(MATH),'rotations_sha256':sha(ROT),'rotations_math_sha256':sha(MATHS),'script_sha256':sha(Path(__file__)),'limitations':'Extracted old Pick function and actual math on synthetic CPU robot/motion objects. These constructed singular configurations may or may not occur in real tasks; no simulator, PhysX, reward-manager configuration or PPO has run. No source changes.'}
with Path(sys.argv[1]).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
