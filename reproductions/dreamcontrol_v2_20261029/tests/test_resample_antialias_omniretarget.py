import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
HAS_SCIPY = importlib.util.find_spec('scipy') is not None
if HAS_SCIPY:
    from resample_antialias_omniretarget import resample_antialias
    from resample_omniretarget import resample


@unittest.skipUnless(HAS_SCIPY, 'requires scipy')
class AntialiasTest(unittest.TestCase):
    def test_reduces_above_nyquist_sinusoid(self):
        t=np.arange(300)/30.; q=np.zeros((len(t),36)); q[:,0]=1
        q[:,7]=np.sin(2*np.pi*12*t)
        raw=resample(q,30,20,max_seconds=None)
        filtered=resample_antialias(q,30,20)
        self.assertEqual(raw.shape,filtered.shape)
        rms=lambda x: np.sqrt(np.mean(x[20:-20,7]**2))
        self.assertLess(rms(filtered),rms(raw)*.2)

    def test_sign_flip_and_duration_are_safe(self):
        q=np.zeros((301,43)); q[:,0]=np.sqrt(.5); q[:,3]=np.sqrt(.5); q[1::2,:4]*=-1
        q[:,4]=np.arange(301)/30.; q[:,36]=1
        out=resample_antialias(q,30,20,max_seconds=10)
        self.assertEqual(out.shape,(200,43))
        np.testing.assert_allclose(np.linalg.norm(out[:,:4],axis=1),1,atol=1e-12)
        np.testing.assert_allclose(np.abs(out[:,0]),np.sqrt(.5),atol=1e-9)
        np.testing.assert_allclose(out[:,4],np.arange(200)/20,atol=.002)
        self.assertEqual(len(resample_antialias(q,30,20)),201)

    def test_rejects_bad_quaternions(self):
        q=np.zeros((4,43)); q[:,0]=1; q[:,36]=.74
        with self.assertRaisesRegex(ValueError,'normalized'):
            resample_antialias(q,30,20)

    def test_coincident_endpoints_are_preserved(self):
        q=np.zeros((301,36)); q[:,0]=1
        q[0,4]=.3; q[1,4]=-.1; q[-1,4]=.7; q[-2,4]=-.2
        out=resample_antialias(q,30,20)
        np.testing.assert_array_equal(out[0],q[0])
        np.testing.assert_array_equal(out[-1],q[-1])

    def test_locked_waist_object_quaternion_and_grid(self):
        q=np.zeros((125,41)); q[:,0]=1; q[:,34]=1
        q[1::2,34:38]*=-1
        q[:,4]=np.arange(len(q))/30.
        out=resample_antialias(q,30,20)
        self.assertEqual(out.shape,(83,41))
        np.testing.assert_allclose(np.linalg.norm(out[:,34:38],axis=1),1,atol=1e-12)
        np.testing.assert_allclose(out[:,4],np.arange(83)/20.,atol=1e-12)
        with self.assertRaisesRegex(ValueError,'normalized'):
            q[:,34:38]*=.5
            resample_antialias(q,30,20)


if __name__ == '__main__':
    unittest.main()
