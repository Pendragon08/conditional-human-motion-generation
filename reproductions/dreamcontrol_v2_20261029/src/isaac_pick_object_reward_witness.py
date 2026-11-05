"""Run old Pick object reset/reward sources against explicit mock states."""
from __future__ import annotations
import ast
import hashlib
import json
import sys
from pathlib import Path
import torch
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
BASE=TRAIN/'source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_interactive_base.py'
PICK=BASE.with_name('motion_tracking_pick_env.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1)
tree=ast.parse(BASE.read_text()); funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('reset_object_state','object_above_threshold')];assert len(funcs)==2
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*funcs],type_ignores=[]);ast.fix_missing_locations(module);ns={'torch':torch};exec(compile(module,str(BASE),'exec'),ns)
class Obj:pass
class FakeObject:
    def __init__(self,n):self.data=Obj();self.data.root_pos_w=torch.zeros((n,3))
    def write_root_pose_to_sim(self,pose,env_ids):self.pose=pose.clone();self.pose_ids=env_ids.clone()
    def write_root_velocity_to_sim(self,v,env_ids):self.velocity=v.clone();self.velocity_ids=env_ids.clone()
class Scene:
    def __init__(self,origins):self.env_origins=origins;self.num_envs=len(origins);self.object=FakeObject(len(origins))
    def __getitem__(self,key):assert key=='object';return self.object
class Library:
    def __init__(self,closed):self.closed=closed
    def get_motion_state(self,ids,times):
        n=len(ids);return {'grab_pos':torch.tensor([.1,.2,.3]).repeat(n,1),'offsets':torch.zeros((n,3)),'is_closed':self.closed[ids]}
origins=torch.tensor([[0.,0.,0.],[2.,0.,.75]])
env=Obj();env.scene=Scene(origins);env.device='cpu';env.num_envs=2;env.start_motion_times=torch.zeros(2);env.motion_ids=torch.arange(2);env.motion_lib=Library(torch.zeros(2,dtype=torch.bool))
ns['reset_object_state'](env,torch.arange(2),height=1.0,offset=[0.,0.]);pose=env.scene.object.pose
assert torch.allclose(pose[:,2],torch.tensor([1.,1.])) and torch.allclose(pose[:,2]-origins[:,2],torch.tensor([1.,.25]))
heights=torch.tensor([.9,.925,.95,.950001,1.]);outputs={}
for closed in (False,True):
    e=Obj();e.num_envs=len(heights);e.device='cpu';e.scene=Scene(torch.zeros((e.num_envs,3)));e.scene.object.data.root_pos_w[:,2]=heights
    e.episode_length_buf=torch.zeros(e.num_envs);e.step_dt=.02;e.start_motion_times=torch.zeros(e.num_envs);e.motion_ids=torch.arange(e.num_envs);e.motion_lib=Library(torch.full((e.num_envs,),closed,dtype=torch.bool));e.n_successes=torch.zeros(e.num_envs)
    v=ns['object_above_threshold'](e,height_thres=.95,fall_thres=.9);outputs[str(closed).lower()]={'reward':v.tolist(),'success_counter_increment':e.n_successes.tolist()}
assert outputs['true']['reward'][2]==0 and outputs['true']['reward'][3]==1 and outputs['true']['reward'][4]==1
result={'base_sha256':sha(BASE),'pick_sha256':sha(PICK),'script_sha256':sha(Path(__file__)),'origins':origins.tolist(),'reset_world_poses':pose.tolist(),'reset_local_heights':(pose[:,2]-origins[:,2]).tolist(),'reset_velocities':env.scene.object.velocity.tolist(),'reward_heights':heights.tolist(),'height_thres':.95,'fall_thres':.9,'outputs':outputs,'limitations':'Old source functions with CPU mock; not an Isaac environment. Nonzero origin and exact equality are synthetic edge cases; objects may move after reset. Reward is height-based and reference-gated, not proof of actual grasp or v2 intent.'}
with Path(sys.argv[1]).open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
