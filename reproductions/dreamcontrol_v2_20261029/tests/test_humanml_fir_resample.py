import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from humanml_fir_resample import fir_resample

class FirTests(unittest.TestCase):
    def test_constant_and_ramp_with_near30_rate(self):
        rate=29.9999942779541;n=121;t=np.arange(n)/rate
        p=np.stack((t,2*t,-t),axis=1);r=np.zeros((n,6));r[:,2]=0.5
        pos,out,meta=fir_resample(p,r,rate)
        expected=np.arange(len(pos))/20
        np.testing.assert_allclose(pos[:,0],expected,atol=1e-8)
        error=(Rotation.from_rotvec(out[:,:3]).inv()*Rotation.from_rotvec(np.tile([0,0,0.5],(len(out),1)))).magnitude()
        self.assertLess(error.max(),1e-8)
        self.assertEqual((meta['up'],meta['down']),(2,3))
        self.assertLessEqual(meta['output_last_time_s'],meta['source_duration_s']+1e-10)

    def test_attenuates_above_output_nyquist(self):
        rate=30.0;n=301;t=np.arange(n)/rate
        p=np.zeros((n,3)); r=np.zeros((n,3));r[:,2]=0.3*np.sin(2*np.pi*12*t)
        filtered_p,filtered,_=fir_resample(p,r,rate)
        # Independent unfiltered 20-Hz evaluations of the analytic 12-Hz rotation.
        target=np.arange(len(filtered))/20.0
        unfiltered=0.3*np.sin(2*np.pi*12*target)
        interior=slice(10,-10)
        self.assertLess(np.std(filtered[interior,2]),np.std(unfiltered[interior])/5)
        np.testing.assert_allclose(filtered_p,0,atol=1e-12)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):fir_resample(np.zeros((3,3)),np.zeros((3,3)),30)

if __name__=='__main__':unittest.main()
