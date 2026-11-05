import unittest
import numpy as np
from omomo_text_motion_triage import observed_motion,review_flags


class MotionTriageTest(unittest.TestCase):
    def test_rise_then_fall_and_quaternion_sign(self):
        obj=np.zeros((5,7));obj[:,0]=[1.,-1.,1.,-1.,1.];obj[:,6]=[0.,.1,.2,.1,0.]
        obj[:,4]=[0.,0.,.05,.10,.10]
        m=observed_motion(obj,np.arange(5)/20.)
        self.assertAlmostEqual(m['ordered_max_rise_m'],.2)
        self.assertAlmostEqual(m['ordered_max_fall_m'],.2)
        self.assertAlmostEqual(m['max_rotation_from_start_rad'],0.)
        flags=review_flags('Lift the largebox, move the largebox, and put down the largebox.',m)
        self.assertEqual(flags,['contact_force_and_support_not_observed'])

    def test_flags_are_not_semantic_acceptance(self):
        obj=np.zeros((3,7));obj[:,0]=1.
        m=observed_motion(obj,np.arange(3)/20.)
        flags=review_flags('Lift the largebox, rotate the largebox, and set it back down.',m)
        self.assertIn('small_observed_center_rise',flags)
        self.assertIn('small_observed_center_fall',flags)
        self.assertIn('small_observed_object_rotation',flags)
        self.assertIn('contact_force_and_support_not_observed',flags)
        obj[0,0]=0.
        with self.assertRaises(ValueError):observed_motion(obj,np.arange(3)/20.)


if __name__=='__main__':unittest.main()
