import importlib.util
import math
import sys
import unittest
from pathlib import Path

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
    from actor_observation import actor_point_observation, actor_object_observation


@unittest.skipUnless(HAS_TORCH, 'requires torch')
class ActorObservationTest(unittest.TestCase):
    def inputs(self):
        return dict(root_pos_w=torch.zeros((1, 3)), root_quat_wxyz=torch.tensor([[1., 0., 0., 0.]]),
                    root_ang_vel_b=torch.zeros((1, 3)), gravity_b=torch.tensor([[0., 0., -1.]]),
                    joint_pos_rel=torch.zeros((1, 27)), joint_vel_rel=torch.zeros((1, 27)),
                    previous_action=torch.zeros((1, 29)), goal_pos_w=torch.tensor([[1., 0., 0.]]))

    def test_root_relative_goal_and_rotation(self):
        args = self.inputs()
        args['root_quat_wxyz'] = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        out = actor_point_observation(**args)
        self.assertEqual(out.shape, (1, 92))
        torch.testing.assert_close(out[0, -3:], torch.tensor([0., -1., 0.]), atol=1e-6, rtol=0)
        delta = torch.tensor([[5., 9., 2.]])
        args['root_pos_w'] += delta
        args['goal_pos_w'] += delta
        torch.testing.assert_close(actor_point_observation(**args), out)

    def test_zero_quaternion_rejected(self):
        args = self.inputs()
        args['root_quat_wxyz'].zero_()
        with self.assertRaisesRegex(ValueError, 'zero root quaternion'):
            actor_point_observation(**args)

    def test_live_object_pose_is_root_relative(self):
        args = self.inputs()
        args['object_pos_w'] = args.pop('goal_pos_w')
        args['root_quat_wxyz'] = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        args['object_quat_wxyz'] = args['root_quat_wxyz'].clone()
        out = actor_object_observation(**args)
        self.assertEqual(out.shape, (1, 96))
        torch.testing.assert_close(out[0, -7:-4], torch.tensor([0., -1., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(out[0, -4:], torch.tensor([1., 0., 0., 0.]), atol=1e-6, rtol=0)
        args['object_quat_wxyz'] *= -1
        torch.testing.assert_close(actor_object_observation(**args), out, atol=1e-6, rtol=0)

    def test_zero_object_quaternion_rejected(self):
        args = self.inputs()
        args['object_pos_w'] = args.pop('goal_pos_w')
        args['object_quat_wxyz'] = torch.zeros((1, 4))
        with self.assertRaisesRegex(ValueError, 'zero object quaternion'):
            actor_object_observation(**args)

    def test_half_turn_quaternion_sign_is_canonical(self):
        args = self.inputs()
        args['object_pos_w'] = args.pop('goal_pos_w')
        args['object_quat_wxyz'] = torch.tensor([[0., 1., 0., 0.]])
        out = actor_object_observation(**args)
        args['object_quat_wxyz'] *= -1
        torch.testing.assert_close(actor_object_observation(**args), out)


if __name__ == '__main__':
    unittest.main()
