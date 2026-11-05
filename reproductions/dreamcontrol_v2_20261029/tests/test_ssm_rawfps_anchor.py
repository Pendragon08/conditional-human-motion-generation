import unittest
import numpy as np
from pyroki_ssm_rawfps_anchor_gpu import build_overlap_anchor


class AnchorTest(unittest.TestCase):
    def setUp(self):
        self.prev = {'indices':np.arange(1,8), 'root':np.arange(49,dtype=np.float32).reshape(7,7),
                     'q':np.arange(189,dtype=np.float32).reshape(7,27)}

    def test_three_frame_overlap(self):
        root,q,mask = build_overlap_anchor(self.prev,np.arange(5,12))
        np.testing.assert_array_equal(mask,[1,1,1,0,0,0,0])
        np.testing.assert_array_equal(root[:3],self.prev['root'][4:])
        np.testing.assert_array_equal(q[:3],self.prev['q'][4:])
        np.testing.assert_array_equal(root[3:,0],np.ones(4))

    def test_six_frame_tail_overlap(self):
        root,q,mask = build_overlap_anchor(self.prev,np.arange(2,9))
        np.testing.assert_array_equal(mask,[1,1,1,1,1,1,0])
        np.testing.assert_array_equal(q[:6],self.prev['q'][1:])

    def test_gap_rejected(self):
        with self.assertRaises(ValueError):
            build_overlap_anchor(self.prev,np.arange(10,17))


if __name__ == '__main__': unittest.main()
