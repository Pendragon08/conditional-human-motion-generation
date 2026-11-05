import unittest
import numpy as np
from render_five_caption_review import interpolate_points


class TimelineTest(unittest.TestCase):
    def test_interpolation_and_source_remain_unchanged(self):
        src = np.array([0., .1, .2])
        points = np.empty((3, 22, 3))
        points[:] = src[:, None, None] * np.array([1., 2., 3.])
        before = points.copy()
        out = interpolate_points(src, points, np.array([.05, .15]))
        np.testing.assert_allclose(out[:, 0], [[.05, .10, .15], [.15, .30, .45]])
        np.testing.assert_array_equal(points, before)

    def test_rejects_extrapolation_and_nonmonotonic_input(self):
        poses = np.zeros((3, 22, 3))
        with self.assertRaises(ValueError):
            interpolate_points([0., .1, .2], poses, [-.01, .1])
        with self.assertRaises(ValueError):
            interpolate_points([0., .1, .1], poses, [0., .1])


if __name__ == '__main__':
    unittest.main()
