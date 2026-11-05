import unittest
import numpy as np
from g1_joint_limit_gate import check_joint_limits, require_joint_limits


class JointLimitGateTest(unittest.TestCase):
    def setUp(self):
        self.names = ['j{}'.format(i) for i in range(27)]
        self.limits = {name: (-1., 1.) for name in self.names}

    def test_bounds_and_explicit_tolerance(self):
        q = np.zeros((3, 27)); q[1, 2] = 1.01
        report = check_joint_limits(q, self.names, self.limits, tolerance_rad=.001)
        self.assertFalse(report['joint_limits_passed'])
        self.assertEqual(report['worst_row'], 1)
        self.assertEqual(report['worst_joint'], 'j2')
        self.assertAlmostEqual(report['max_excess_rad'], .01)
        with self.assertRaises(ValueError):
            require_joint_limits(q, self.names, self.limits, tolerance_rad=.001)
        self.assertTrue(check_joint_limits(q, self.names, self.limits, tolerance_rad=.02)['joint_limits_passed'])
        self.assertFalse(check_joint_limits(q, self.names, self.limits, tolerance_rad=.02)['training_ready'])

    def test_reject_bad_shapes_names_and_tolerance(self):
        q = np.zeros((2, 27))
        for names in (self.names[:-1], self.names[:-1] + [self.names[0]]):
            with self.assertRaises(ValueError):
                check_joint_limits(q, names, self.limits, tolerance_rad=0.)
        with self.assertRaises(ValueError):
            check_joint_limits(q, self.names, self.limits, tolerance_rad=float('nan'))
        q[0, 0] = np.nan
        with self.assertRaises(ValueError):
            check_joint_limits(q, self.names, self.limits, tolerance_rad=0.)


if __name__ == '__main__':
    unittest.main()
