import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from omniretarget_bridge import convert_layout


class OmniRetargetBridgeTest(unittest.TestCase):
    def test_reorders_base_and_preserves_joints_and_object(self):
        q = np.zeros((2, 43))
        q[:, :4] = [0.5, 0.5, 0.5, 0.5]
        q[:, 4:7] = [2, 3, 4]
        q[:, 7:36] = np.arange(29)
        q[:, 36] = 1
        q[:, 40:] = [5, 6, 7]
        original = q.copy()
        robot, obj = convert_layout(q)
        np.testing.assert_array_equal(robot[:, :3], q[:, 4:7])
        np.testing.assert_array_equal(robot[:, 3:7], q[:, [1, 2, 3, 0]])
        np.testing.assert_array_equal(robot[:, 7:], q[:, 7:36])
        np.testing.assert_array_equal(obj, q[:, 36:43])
        np.testing.assert_array_equal(q, original)

    def test_rejects_invalid_quaternions(self):
        q = np.zeros((2, 36))
        with self.assertRaisesRegex(ValueError, 'zero base'):
            convert_layout(q)
        q[:, 0] = 2
        with self.assertRaisesRegex(ValueError, 'normalized'):
            convert_layout(q)


if __name__ == '__main__':
    unittest.main()
