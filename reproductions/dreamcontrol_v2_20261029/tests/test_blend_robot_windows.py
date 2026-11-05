import unittest
import numpy as np
from blend_robot_windows import blend_windows

def window(start,pos,sign=1):
    root=np.zeros((7,7));root[:,0]=sign;root[:,4]=pos
    return {'frames':np.arange(start,start+7),'root_wxyz_xyz':root,'joint_pos':np.full((7,27),pos),'joint_names':[str(i) for i in range(27)]}

class BlendTest(unittest.TestCase):
    def test_weights_and_quaternion_signs(self):
        out=blend_windows([window(0,0),window(4,1,-1)])
        self.assertEqual(len(out['frames']),11)
        self.assertAlmostEqual(out['blended'][4,4],1/4)
        self.assertAlmostEqual(out['blended'][6,4],3/4)
        np.testing.assert_allclose(out['blended'][:,:4],np.tile([1.,0,0,0],(11,1)))
    def test_reject_gaps_and_joint_order(self):
        with self.assertRaises(ValueError):blend_windows([window(0,0),window(8,1)])
        other=window(4,1);other['joint_names'].reverse()
        with self.assertRaises(ValueError):blend_windows([window(0,0),other])

if __name__=='__main__':unittest.main()
