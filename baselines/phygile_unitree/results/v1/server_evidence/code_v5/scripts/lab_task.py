"""PhyGile custom task hosted in Unitree RL Lab, with its paper-specific MDP.

The Unitree default dance task is a structural parent, not a replacement policy.
Import only after Isaac Sim AppLauncher. No vendor or parent file is modified.
"""
import copy
from isaaclab.utils import configclass
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg import RobotEnvCfg as UnitreeG1MimicCfg
from unitree_rl_lab.assets.robots.unitree import UnitreeArticulationCfg
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatWoStateEstimationEnvCfg

@configclass
class PhyGileUnitreeEnvCfg(UnitreeG1MimicCfg):
    def __post_init__(self):
        super().__post_init__()
        sdk_names=list(self.scene.robot.joint_sdk_names)
        paper=G1FlatWoStateEstimationEnvCfg()
        for key,value in vars(paper).items():
            if not key.startswith('_'):setattr(self,key,copy.deepcopy(value))
        values={key:copy.deepcopy(value) for key,value in vars(paper.scene.robot).items() if not key.startswith('_')}
        self.scene.robot=UnitreeArticulationCfg(**values,joint_sdk_names=sdk_names)

def semantic_configuration(cfg):
    """Exact MDP comparison; exclude only Unitree's extra SDK order metadata."""
    value=cfg.to_dict()
    value['scene']['robot'].pop('joint_sdk_names',None)
    return {key:value[key] for key in ['scene','observations','actions','commands','rewards','terminations','events','sim','decimation','episode_length_s']}

def register():
    import gymnasium as gym
    if 'Unitree-G1-PhyGile-v0' not in gym.registry:
        gym.register('Unitree-G1-PhyGile-v0',entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True,
          kwargs={'env_cfg_entry_point':'lab_task:PhyGileUnitreeEnvCfg',
                  'rsl_rl_cfg_entry_point':'whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg'})
