"""Exercise Isaac Lab source ObservationManager with mocked Sim/assets.

No Isaac Sim is started. Isaac configclass, SceneEntityCfg, manager configs,
ManagerBase and ObservationManager are loaded unmodified from local source;
Omniverse, simulation, asset and task-parent interfaces are in-process stubs.
This is not a real Isaac Lab/PhysX environment test.
"""
import hashlib,importlib,json,sys,types
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
STAGE=ROOT/'isaac_live_actor_adapter_v2_20261030';sys.path.insert(0,str(STAGE/'src'));sys.path.insert(0,str(STAGE/'tests'))
ISAAC=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/source/isaaclab/isaaclab')

def package(name,path=None):
    module=types.ModuleType(name);module.__path__=[] if path is None else [str(path)];sys.modules[name]=module;return module

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    package('isaaclab',ISAAC);utils=package('isaaclab.utils',ISAAC/'utils')
    warp=package('warp');warp.array=type('WarpArray',(),{});warp.torch=SimpleNamespace()
    strings=importlib.import_module('isaaclab.utils.string');dictionary=importlib.import_module('isaaclab.utils.dict')
    utils.configclass=importlib.import_module('isaaclab.utils.configclass').configclass
    utils.class_to_dict=dictionary.class_to_dict;utils.string_to_callable=strings.string_to_callable
    mods=package('isaaclab.utils.modifiers');mods.ModifierCfg=type('ModifierCfg',(),{});mods.ModifierBase=type('ModifierBase',(),{});utils.modifiers=mods
    noise=package('isaaclab.utils.noise');noise.NoiseCfg=type('NoiseCfg',(),{})
    buffers=package('isaaclab.utils.buffers',ISAAC/'utils/buffers');buffers.CircularBuffer=importlib.import_module('isaaclab.utils.buffers.circular_buffer').CircularBuffer
    assets=package('isaaclab.assets')
    for n in ('Articulation','RigidObject','RigidObjectCollection'):setattr(assets,n,type(n,(),{}))
    scene=package('isaaclab.scene');scene.InteractiveScene=type('InteractiveScene',(),{})
    omni=package('omni');omni.log=package('omni.log');omni.log.info=lambda *args:None
    omni.timeline=package('omni.timeline')
    pt=package('prettytable');pt.PrettyTable=type('PrettyTable',(),{})
    managers=package('isaaclab.managers',ISAAC/'managers')
    entity=importlib.import_module('isaaclab.managers.scene_entity_cfg');term=importlib.import_module('isaaclab.managers.manager_term_cfg')
    managers.SceneEntityCfg=entity.SceneEntityCfg;managers.ObservationTermCfg=term.ObservationTermCfg;managers.ObservationGroupCfg=term.ObservationGroupCfg
    manager=importlib.import_module('isaaclab.managers.observation_manager')
    for name in ('isaaclab_tasks','isaaclab_tasks.manager_based','isaaclab_tasks.manager_based.interactive_motion_tracking','isaaclab_tasks.manager_based.interactive_motion_tracking.g1'):
        package(name)
    task=package('isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_pick_env')
    @utils.configclass
    class MockG1PickEnvCfg:
        observations: object=None
    task.G1PickEnvCfg=MockG1PickEnvCfg
    import isaac_live_pick_cfg
    from test_isaac_live_actor import env_and_configs
    cfg=isaac_live_pick_cfg.G1PickLiveObjectEnvCfg()
    env,_,_=env_and_configs();env.num_envs=2;env.device='cpu';env.sim=SimpleNamespace(is_playing=lambda:True)
    robot=env.scene['robot'];robot.num_joints=len(robot.joint_names)
    def find_joints(names,preserve_order=False):
        if not preserve_order:raise AssertionError('joint order was not preserved')
        return [robot.joint_names.index(n) for n in names],list(names)
    robot.find_joints=find_joints
    obs=manager.ObservationManager(cfg.observations,env)
    result=obs.compute()['policy']
    if result.shape!=(2,96):raise ValueError('unexpected output shape')
    old=result.clone();env.motion_ids=torch.tensor([1,2]);env.start_motion_times=torch.tensor([3.,4.])
    torch.testing.assert_close(old,obs.compute()['policy'])
    env.scene['object'].data.root_pos_w[:,0]+=2.
    new=obs.compute()['policy'];torch.testing.assert_close(new[:,-7],torch.full((2,),3.))
    resolved=obs._group_obs_term_cfgs['policy'][0].params['robot_cfg'].joint_ids
    if list(resolved)!=list(range(27)):raise ValueError('joint resolution mismatch')
    source_paths=[ISAAC/'utils/configclass.py',ISAAC/'managers/scene_entity_cfg.py',ISAAC/'managers/manager_term_cfg.py',ISAAC/'managers/manager_base.py',ISAAC/'managers/observation_manager.py',STAGE/'src/isaac_live_actor.py',STAGE/'src/isaac_live_pick_cfg.py',Path(__file__)]
    report={'source_hashes':{p.name:sha(p) for p in source_paths},'policy_shape':list(result.shape),'resolved_joint_count':len(resolved),
            'reference_mutation_invariant':True,'live_object_motion_detected':True,
            'limitations':'Actual Isaac Lab manager/configclass/SceneEntityCfg source ran in process with mocked simulation, assets, task parent, Omni, warp and prettytable. No Isaac Sim, real manager lifecycle, PhysX, task rewards, policy or PPO were run; manager source behavior may differ in its real environment.'}
    with Path(__file__).with_name('report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
