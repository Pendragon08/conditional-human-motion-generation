import unittest
import numpy as np
from project_joint_limits import project


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.names = ['j%d' % i for i in range(27)]
        self.limits = {name: (-1., 1.) for name in self.names}

    def test_exact_projection_and_unchanged_valid_values(self):
        q = np.zeros((3, 27)); q[0, 0] = 1.0002; q[1, 4] = -1.0001
        fixed, before, after = project(q, self.names, self.limits, .001)
        self.assertFalse(before['joint_limits_passed'])
        self.assertTrue(after['joint_limits_passed'])
        self.assertEqual(fixed[0, 0], 1.)
        self.assertEqual(fixed[1, 4], -1.)
        self.assertEqual(np.count_nonzero(fixed != q), 2)
        self.assertEqual(q[0, 0], 1.0002)

    def test_reject_large_violation_and_bad_joint_mapping(self):
        q = np.zeros((3, 27)); q[0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            project(q, self.names, self.limits, .001)
        q[0, 0] = 0
        with self.assertRaises(ValueError):
            project(q, self.names[:-1] + [self.names[0]], self.limits, .001)


if __name__ == '__main__': unittest.main()
