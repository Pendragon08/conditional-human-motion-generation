import unittest
from pathlib import Path
import numpy as np
import torch
from motion_bridge import SOURCE_NAMES
from waist_lock_ik import _chain, _positions, _joint_limits
from human_keypoint_robot_fit import fit_human_keypoints, HUMAN_KEYPOINT_INDICES

URDF=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')


class HumanFitTest(unittest.TestCase):
    def test_synthetic_reachable_target_reduces_error_and_keeps_limits(self):
        torch.set_num_threads(1)
        chain=_chain(URDF); names=chain.get_joint_parameter_names()
        ordered=[n for n in SOURCE_NAMES if n not in ('waist_roll_joint','waist_pitch_joint')]
        q=np.zeros((4,27)); q[:,names.index('left_knee_joint')]=.45
        q[:,names.index('right_knee_joint')]=.3
        q[:,names.index('left_elbow_joint')]=.5
        points=_positions(chain,torch.tensor(q,dtype=torch.float64)).detach().numpy()
        human=np.zeros((4,22,3)); human[:,HUMAN_KEYPOINT_INDICES]=points
        initial=np.zeros((4,34)); initial[:,6]=1.
        fit=fit_human_keypoints(human,initial,URDF,leg_scale=1.,arm_scale=1.,pelvis_scale=1.,
              steps=120,direction_weight=0.,foot_velocity_weight=0.,nominal_weight=0.,temporal_weight=0.)
        self.assertLess(fit['final_mean_error_m'],fit['initial_mean_error_m']*.5)
        lo,hi=_joint_limits(URDF,list(fit['joint_names']))
        self.assertTrue(np.all(fit['joint_pos'] >= lo.numpy()-1e-10))
        self.assertTrue(np.all(fit['joint_pos'] <= hi.numpy()+1e-10))
        self.assertTrue(np.allclose(np.linalg.norm(fit['root_quat_xyzw'],axis=1),1.))

    def test_rejects_unaligned_and_invalid_data(self):
        human=np.zeros((4,22,3)); motion=np.zeros((3,34)); motion[:,6]=1.
        with self.assertRaises(ValueError): fit_human_keypoints(human,motion,URDF)
        motion=np.zeros((4,34)); motion[:,6]=1.; human[0,0,0]=np.nan
        with self.assertRaises(ValueError): fit_human_keypoints(human,motion,URDF)


if __name__=='__main__': unittest.main()
