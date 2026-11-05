import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from feature263 import canonicalize_z_up


class CanonicalizationTest(unittest.TestCase):
    def test_root_origin_floor_and_facing(self):
        p = np.zeros((3, 22, 3))
        p[..., 2] = 1
        p[:, 1, 1] = -0.2
        p[:, 2, 1] = 0.2
        p[:, 16, 1] = -0.3
        p[:, 17, 1] = 0.3
        p[:, 10, 2] = 0
        result = canonicalize_z_up(p)
        self.assertAlmostEqual(float(result[..., 1].min()), 0)
        np.testing.assert_allclose(result[0, 0, [0, 2]], 0, atol=1e-12)
        across = result[0, 2] - result[0, 1] + result[0, 17] - result[0, 16]
        forward = np.cross([0, 1, 0], across)
        np.testing.assert_allclose(forward[[0, 2]] / np.linalg.norm(forward), [0, 1], atol=1e-12)

    def test_invalid_points_are_rejected(self):
        with self.assertRaises(ValueError):
            canonicalize_z_up(np.zeros((2, 22, 3)))


if __name__ == "__main__":
    unittest.main()
