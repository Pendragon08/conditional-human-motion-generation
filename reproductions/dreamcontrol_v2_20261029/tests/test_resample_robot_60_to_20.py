import unittest
import numpy as np
from resample_robot_60_to_20 import resample_robot_60_to_20


class ResampleTest(unittest.TestCase):
    def test_strict_grid_no_extrapolation_and_constant_motion(self):
        fps=59.9999885559082
        times=26/fps+np.arange(399)/fps
        state=np.zeros((len(times),34)); state[:,0]=1.; state[:,4]=1.2+2*(times-times[0])
        result=resample_robot_60_to_20(state,times)
        np.testing.assert_allclose(np.diff(result['times']),.05,rtol=0,atol=1e-12)
        self.assertLessEqual(result['times'][-1],times[-1])
        np.testing.assert_allclose(result['states'][:,4],1.2+2*(result['times']-times[0]),atol=1e-10,rtol=0)
        np.testing.assert_allclose(np.linalg.norm(result['states'][:,:4],axis=1),1.,atol=1e-10,rtol=0)

    def test_antialias_for_25hz_synthetic_joint_signal(self):
        fps=60.; times=np.arange(601)/fps
        state=np.zeros((len(times),34)); state[:,0]=1.; state[:,7]=np.sin(2*np.pi*25*times)
        out=resample_robot_60_to_20(state,times)['states'][:,7]
        self.assertLess(np.sqrt(np.mean(out[10:-10]**2)), .05)
        self.assertGreater(np.sqrt(np.mean(state[::3,7][10:-10]**2)),.5)

    def test_rejects_wrong_rate_and_bad_quaternion(self):
        state=np.zeros((20,34)); state[:,0]=1.
        with self.assertRaises(ValueError): resample_robot_60_to_20(state,np.arange(20)/30)
        state[:,0]=0.
        with self.assertRaises(ValueError): resample_robot_60_to_20(state,np.arange(20)/60)


if __name__ == '__main__': unittest.main()
