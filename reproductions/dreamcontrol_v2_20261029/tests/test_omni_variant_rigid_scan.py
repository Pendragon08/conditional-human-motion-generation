import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from omni_variant_rigid_scan import compare


class VariantTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.a = np.zeros((8,43))
        self.a[:, 0] = self.a[:, 36] = 1.
        self.a[:, 4:7] = rng.normal(size=(8,3))
        self.a[:, 7:36] = rng.normal(size=(8,29))
        self.a[:, 40:43] = rng.normal(size=(8,3))

    def test_common_yaw_translation_and_quaternion_sign(self):
        r = Rotation.from_euler('z', .7)
        b = self.a.copy()
        b[:, 4:7] = r.apply(self.a[:, 4:7]) + [1., 2., 0.]
        b[:, 40:43] = r.apply(self.a[:, 40:43]) + [1., 2., 0.]
        wxyz = r.as_quat()[[3,0,1,2]]
        b[:, :4] = wxyz
        b[:, 36:40] = -wxyz
        self.assertEqual(compare(self.a,b)['status'], 'planar_rigid_within_numeric_tolerance')

    def test_different_object_or_joint_is_not_a_rigid_variant(self):
        b = self.a.copy()
        b[3, 40] += .1
        self.assertEqual(compare(self.a,b)['status'], 'not_planar_rigid_within_numeric_tolerance')
        b = self.a.copy()
        b[1, 7] += .01
        self.assertEqual(compare(self.a,b)['status'], 'not_planar_rigid_within_numeric_tolerance')

    def test_vertical_shift_or_bad_shape_is_not_accepted(self):
        b = self.a.copy()
        b[:, 6] += .1
        b[:, 42] += .1
        self.assertEqual(compare(self.a,b)['status'], 'not_planar_rigid_within_numeric_tolerance')
        self.assertEqual(compare(self.a,b[:-1])['status'], 'shape_mismatch')


if __name__ == '__main__':
    unittest.main()
