"""Experimental opt-in Pick config; not registered or tested in Isaac Lab.

The inherited task scene/rewards/resets remain from DreamControl v1, not v2.
This changes the actor input layout and is incompatible with old checkpoints.
Do not train until the actual environment and dataset gates are validated.
"""
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_pick_env import G1PickEnvCfg

from isaac_live_actor import G1_27_JOINT_NAMES, live_object_policy_observation


@configclass
class LiveObjectPolicyCfg(ObsGroup):
    live_state = ObsTerm(
        func=live_object_policy_observation,
        params={
            'robot_cfg': SceneEntityCfg('robot', joint_names=list(G1_27_JOINT_NAMES), preserve_order=True),
            'object_cfg': SceneEntityCfg('object'),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class LiveObjectObservationsCfg:
    policy: LiveObjectPolicyCfg = LiveObjectPolicyCfg()


@configclass
class G1PickLiveObjectEnvCfg(G1PickEnvCfg):
    observations: LiveObjectObservationsCfg = LiveObjectObservationsCfg()
