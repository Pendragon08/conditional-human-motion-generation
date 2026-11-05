import unittest
import numpy as np
from project_vertical_support_bound import project_vertical_support


class VerticalProjectionTests(unittest.TestCase):
    def test_rest_unchanged(self):
        t=np.arange(20)/20;c=np.zeros((20,3))
        r=project_vertical_support(t,c,[0,0,-7.5],max_shift_m=.02,smoothness_weight=10)
        np.testing.assert_allclose(r['height_shift_m'],0,atol=1e-10)
        self.assertFalse(r['training_ready'])

    def test_local_impossible_curvature_can_be_corrected(self):
        t=np.arange(20)/20;c=np.zeros((20,3));c[10,2]=.02
        r=project_vertical_support(t,c,[0,0,-7.5],max_shift_m=.02,smoothness_weight=10)
        self.assertLess(r['before_n_per_kg'].min(),0);self.assertGreaterEqual(r['after_n_per_kg'].min(),-1e-6)
        self.assertLessEqual(np.max(np.abs(r['height_shift_m'])),.02+1e-10)
        self.assertEqual(r['height_shift_m'][0],0);self.assertEqual(r['height_shift_m'][-1],0)

    def test_insufficient_shift_rejected(self):
        t=np.arange(20)/20;c=np.zeros((20,3));c[10,2]=1.
        with self.assertRaises(ValueError):project_vertical_support(t,c,[0,0,-7.5],max_shift_m=.001,smoothness_weight=10)


if __name__=='__main__':unittest.main()
