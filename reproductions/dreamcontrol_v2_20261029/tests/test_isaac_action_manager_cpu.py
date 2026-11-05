"""Actual Isaac action-manager source with AST-extracted v1 config and mock assets.

CPU integration witness only; no Isaac Sim/PhysX, real USD asset, policy or PPO.
"""
import ast,hashlib,importlib,json,sys,types
from dataclasses import MISSING
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE=ROOT/'isaac_live_actor_adapter_v2_20261030';sys.path.insert(0,str(STAGE/'src'))
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/source')
ISAAC=TRAIN/'isaaclab/isaaclab';TASK=TRAIN/'isaaclab_tasks/isaaclab_tasks'
def package(name,path=None):
    m=types.ModuleType(name);m.__path__=[] if path is None else [str(path)];sys.modules[name]=m;return m
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def classes(path,names,ns):
    tree=ast.parse(path.read_text());selected=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
    if {n.name for n in selected}!=set(names):raise ValueError('class extraction mismatch')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
def main():
    package('isaaclab',ISAAC);utils=package('isaaclab.utils',ISAAC/'utils')
    warp=package('warp');warp.array=type('WarpArray',(),{});warp.torch=SimpleNamespace()
    strings=importlib.import_module('isaaclab.utils.string');dictionary=importlib.import_module('isaaclab.utils.dict')
    utils.configclass=importlib.import_module('isaaclab.utils.configclass').configclass;utils.class_to_dict=dictionary.class_to_dict;utils.string_to_callable=strings.string_to_callable
    mods=package('isaaclab.utils.modifiers');mods.ModifierCfg=type('ModifierCfg',(),{});mods.ModifierBase=type('ModifierBase',(),{});utils.modifiers=mods
    noise=package('isaaclab.utils.noise');noise.NoiseCfg=type('NoiseCfg',(),{})
    assets=package('isaaclab.assets');assets.AssetBase=type('AssetBase',(),{})
    for n in ('Articulation','RigidObject','RigidObjectCollection'):setattr(assets,n,type(n,(),{}))
    art=package('isaaclab.assets.articulation');art.Articulation=assets.Articulation
    scene=package('isaaclab.scene');scene.InteractiveScene=type('InteractiveScene',(),{})
    omni=package('omni');omni.log=package('omni.log');omni.log.info=lambda *a:None;omni.timeline=package('omni.timeline');omni.kit=package('omni.kit');omni.kit.app=package('omni.kit.app')
    pt=package('prettytable');pt.PrettyTable=type('PrettyTable',(),{})
    package('isaaclab.managers',ISAAC/'managers');term=importlib.import_module('isaaclab.managers.manager_term_cfg');manager=importlib.import_module('isaaclab.managers.action_manager')
    package('isaaclab.envs',ISAAC/'envs');package('isaaclab.envs.mdp',ISAAC/'envs/mdp');package('isaaclab.envs.mdp.actions',ISAAC/'envs/mdp/actions')
    joint=importlib.import_module('isaaclab.envs.mdp.actions.joint_actions');binary=importlib.import_module('isaaclab.envs.mdp.actions.binary_joint_actions')
    cfgpath=ISAAC/'envs/mdp/actions/actions_cfg.py';basepath=TASK/'manager_based/interactive_motion_tracking/g1/motion_tracking_interactive_base.py';pickpath=TASK/'manager_based/interactive_motion_tracking/g1/motion_tracking_pick_env.py';motionpath=TASK/'utils/motion_lib/motion_lib_base.py'
    ns=package('action_witness_cfg').__dict__;ns.update({'configclass':utils.configclass,'ActionTermCfg':term.ActionTermCfg,'ActionTerm':manager.ActionTerm,'joint_actions':joint,'binary_joint_actions':binary,'MISSING':MISSING})
    classes(cfgpath,{'JointActionCfg','JointPositionActionCfg','BinaryJointActionCfg','BinaryJointPositionActionCfg'},ns)
    mdp=SimpleNamespace(JointPositionActionCfg=ns['JointPositionActionCfg'],BinaryJointPositionActionCfg=ns['BinaryJointPositionActionCfg'])
    tree=ast.parse(motionpath.read_text());matches=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JointNamesOrder' for t in n.targets)]
    assert len(matches)==1;order=ast.literal_eval(matches[0])
    from isaac_live_actor import G1_27_JOINT_NAMES,live_object_policy_observation
    assert tuple(order)==G1_27_JOINT_NAMES
    cfg=package('action_witness_task').__dict__;cfg.update({'configclass':utils.configclass,'mdp':mdp,'JointNamesOrder':order,'np':np})
    classes(basepath,{'ActionsCfg'},cfg);cfg['ActionsCfgBase']=cfg['ActionsCfg'];classes(pickpath,{'ActionsCfg'},cfg)
    cfgobj=cfg['ActionsCfg']()
    finger_names=[f'{side}_hand_{finger}_{idx}_joint' for side in ('left','right') for finger,indices in (('index',(0,1)),('middle',(0,1)),('thumb',(0,1,2))) for idx in indices]
    class Robot(assets.Articulation):
        def __init__(self):
            self.joint_names=order+finger_names;self.num_joints=len(self.joint_names);B=2
            self.data=SimpleNamespace(default_joint_pos=torch.zeros(B,self.num_joints),default_joint_vel=torch.zeros(B,self.num_joints),joint_pos=torch.zeros(B,self.num_joints),joint_vel=torch.zeros(B,self.num_joints),root_pos_w=torch.zeros(B,3),root_quat_w=torch.tensor([[1.,0,0,0]]*B),root_ang_vel_b=torch.zeros(B,3),projected_gravity_b=torch.tensor([[0.,0,-1.]]*B));self.calls=[]
        def find_joints(self,query,preserve_order=False):return strings.resolve_matching_names(query,self.joint_names,preserve_order=preserve_order)
        def set_joint_position_target(self,value,joint_ids):self.calls.append((value.clone(),joint_ids))
    robot=Robot();obj=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[1.,2.,3.]]*2),root_quat_w=torch.tensor([[1.,0,0,0]]*2)))
    env=SimpleNamespace(scene={'robot':robot,'object':obj},num_envs=2,device='cpu',sim=SimpleNamespace(is_playing=lambda:True))
    act=manager.ActionManager(cfgobj,env);env.action_manager=act
    if act.active_terms!=['joint_pos','left_hand_action','right_hand_action'] or act.action_term_dim!=[27,1,1]:raise ValueError('unexpected action ordering or dimensions')
    action=torch.zeros(2,29);action[:,0]=.4;action[0,27:]=torch.tensor([1.,1.]);action[1,27:]=torch.tensor([-1.,-1.]);act.process_action(action);act.apply_action()
    torch.testing.assert_close(act.action,action);assert len(robot.calls)==3
    torch.testing.assert_close(robot.calls[0][0][:,0],torch.tensor([.2,.2]));torch.testing.assert_close(robot.calls[1][0][0],robot.calls[1][0][1]);assert not torch.equal(robot.calls[2][0][0],robot.calls[2][0][1])
    rc=SimpleNamespace(name='robot',joint_ids=list(range(27)));oc=SimpleNamespace(name='object')
    obs=live_object_policy_observation(env,robot_cfg=rc,object_cfg=oc);assert obs.shape==(2,96)
    torch.testing.assert_close(obs[:,60:89],action)
    env.motion_ids=torch.tensor([2,7]);env.start_motion_times=torch.tensor([1.,5.]);torch.testing.assert_close(obs,live_object_policy_observation(env,robot_cfg=rc,object_cfg=oc))
    act.process_action(-action);torch.testing.assert_close(act.prev_action,action);torch.testing.assert_close(act.action,-action)
    paths=[ISAAC/'utils/configclass.py',ISAAC/'utils/string.py',ISAAC/'managers/manager_base.py',ISAAC/'managers/manager_term_cfg.py',ISAAC/'managers/action_manager.py',ISAAC/'envs/mdp/actions/joint_actions.py',ISAAC/'envs/mdp/actions/binary_joint_actions.py',cfgpath,basepath,pickpath,motionpath,STAGE/'src/isaac_live_actor.py',STAGE/'src/actor_observation.py',Path(__file__)]
    result={'source_hashes':{str(p):sha(p) for p in paths},'terms':act.active_terms,'action_dimensions':act.action_term_dim,'total_actions':act.total_action_dim,'observation_shape':list(obs.shape),'action_in_observation_matches':True,
            'left_open_and_close_equal':True,'right_open_and_close_differ':True,'reference_mutation_invariant':True,
            'limitations':'Actual manager and action implementation source with AST-extracted v1 config classes, synthetic articulation, joint names and defaults; no actual USD, Isaac Sim, PhysX, reward, reset, policy or PPO. Left identical commands are inherited v1 configuration; intended semantics and v2 match unproven.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
