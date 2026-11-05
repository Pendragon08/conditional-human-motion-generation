import unittest
import numpy as np
from vertical_support_bound import vertical_support_bound


class VerticalBoundTests(unittest.TestCase):
    def test_freefall_and_rest(self):
        t=np.arange(20)/20;free=np.zeros((20,3));free[:,2]=-0.5*9.81*t*t
        r=vertical_support_bound(t,free,[0,0,-9.81])
        np.testing.assert_allclose(r['required_vertical_support_n_per_kg'],0,atol=1e-10)
        r=vertical_support_bound(t,np.zeros_like(free),[0,0,-9.81])
        np.testing.assert_allclose(r['required_vertical_support_n_per_kg'],9.81,atol=1e-10)
        self.assertFalse(r['training_ready'])

    def test_faster_downward_is_negative(self):
        t=np.arange(7)/20;p=np.zeros((7,3));p[:,2]=-6*t*t
        r=vertical_support_bound(t,p,[0,0,-9.81]);self.assertEqual(r['negative_vertical_support_samples'],5)
        np.testing.assert_allclose(r['required_vertical_support_n_per_kg'],-2.19,atol=1e-10)

    def test_bad_time_gravity_and_nonfinite_rejected(self):
        t=np.array([0.,.05,.11]);p=np.zeros((3,3))
        with self.assertRaises(ValueError):vertical_support_bound(t,p,[0,0,-9.81])
        with self.assertRaises(ValueError):vertical_support_bound([0,.05,.10],p,[1,0,-9.81])
        p[1,0]=np.nan
        with self.assertRaises(ValueError):vertical_support_bound([0,.05,.10],p,[0,0,-9.81])


if __name__=='__main__':unittest.main()
