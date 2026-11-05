import unittest
import numpy as np
from resample_g1_27 import resample_27

class ResampleTest(unittest.TestCase):
    def test_timing_linear_values_and_quaternion_sign(self):
        q=np.zeros((121,34));q[:,0]=1;q[::2,:4]*=-1
        q[:,4]=np.arange(121)/120.;q[:,7]=2*np.arange(121)/120.
        out=resample_27(q)
        self.assertEqual(len(out),21)
        np.testing.assert_allclose(out[:,4],np.arange(21)/20.,atol=1e-10)
        np.testing.assert_allclose(out[:,7],2*np.arange(21)/20.,atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(out[:,:4],axis=1),1,atol=1e-12)
    def test_rejects_invalid_data(self):
        q=np.zeros((12,34))
        with self.assertRaises(ValueError):resample_27(q)
        q[:,0]=1;q[0,7]=np.nan
        with self.assertRaises(ValueError):resample_27(q)
    def test_high_frequency_attenuation(self):
        n=1201;q=np.zeros((n,34));q[:,0]=1
        q[:,7]=np.sin(2*np.pi*28*np.arange(n)/120)
        out=resample_27(q)
        self.assertLess(np.max(np.abs(out[10:-10,7])),.01)

if __name__=='__main__':unittest.main()
