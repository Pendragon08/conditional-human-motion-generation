"""CPU contract tests with fake Isaac data; not an Isaac Lab runtime test."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from isaac_live_actor import G1_27_JOINT_NAMES, live_object_policy_observation


class NoReferenceEnv:
    @property
    def motion_lib(self):
        raise AssertionError('actor must not access the reference motion')


def env_and_configs():
    batch = 2
    robot = SimpleNamespace(joint_names=list(G1_27_JOINT_NAMES) + ['hand_joint'])
    robot.data = SimpleNamespace(
        root_pos_w=torch.tensor([[1.,2.,3.],[4.,5.,6.]]),
        root_quat_w=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]]),
        root_ang_vel_b=torch.zeros(batch,3), projected_gravity_b=torch.tensor([[0.,0.,-1.]]*batch),
        joint_pos=torch.arange(28,dtype=torch.float32).repeat(batch,1),
        default_joint_pos=torch.ones(batch,28), joint_vel=torch.ones(batch,28)*2,
        default_joint_vel=torch.ones(batch,28)*.5,
    )
    obj = SimpleNamespace(data=SimpleNamespace(root_pos_w=robot.data.root_pos_w+torch.tensor([1.,0.,0.]),
                                               root_quat_w=torch.tensor([[1.,0.,0.,0.]]*batch)))
    env=NoReferenceEnv();env.scene={'robot':robot,'object':obj}
    env.action_manager=SimpleNamespace(action=torch.zeros(batch,29))
    return env,SimpleNamespace(name='robot',joint_ids=list(range(27))),SimpleNamespace(name='object')


class AdapterTest(unittest.TestCase):
    def test_live_fields_and_reference_independence(self):
        env,robot,obj=env_and_configs()
        result=live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj)
        self.assertEqual(result.shape,(2,96))
        torch.testing.assert_close(result[:,6:33],env.scene['robot'].data.joint_pos[:,:27]-1)
        torch.testing.assert_close(result[:,-7:-4],torch.tensor([[1.,0.,0.]]*2))
        torch.testing.assert_close(result[:,-4:],torch.tensor([[1.,0.,0.,0.]]*2))
        # Arbitrary reference changes must not affect live observations.
        env.motion_ids=torch.tensor([100,200]);env.start_motion_times=torch.tensor([9.,99.])
        torch.testing.assert_close(result,live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj))
        env.scene['object'].data.root_pos_w[:,0]+=2
        changed=live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj)
        torch.testing.assert_close(changed[:,-7],torch.full((2,),3.))

    def test_reject_unresolved_or_misordered_joints(self):
        env,robot,obj=env_and_configs();robot.joint_ids=slice(None)
        with self.assertRaises(ValueError):live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj)
        robot.joint_ids=list(range(27));robot.joint_ids[0],robot.joint_ids[1]=robot.joint_ids[1],robot.joint_ids[0]
        with self.assertRaises(ValueError):live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj)

    def test_reject_invalid_action_and_pose(self):
        env,robot,obj=env_and_configs();env.action_manager.action=torch.zeros(2,28)
        with self.assertRaises(ValueError):live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj)
        env.action_manager.action=torch.zeros(2,29);env.scene['object'].data.root_quat_w[0]=0
        with self.assertRaises(ValueError):live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj)

    def test_isaac_full_joint_slice_only_when_order_is_proven(self):
        env,robot,obj=env_and_configs();env.scene['robot'].joint_names=list(G1_27_JOINT_NAMES)
        robot.joint_ids=slice(None);robot.joint_names=list(G1_27_JOINT_NAMES)
        self.assertEqual(live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj).shape,(2,96))
        robot.joint_names=list(reversed(robot.joint_names))
        with self.assertRaises(ValueError):live_object_policy_observation(env,robot_cfg=robot,object_cfg=obj)

if __name__=='__main__':unittest.main()
