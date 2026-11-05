import unittest
import numpy as np
from pyroki_ssm_new6_anchor_gpu import anchor_for,window_starts

class AnchorTest(unittest.TestCase):
    def test_first_window_has_no_active_anchor(self):
        root,q,mask=anchor_for(None,np.arange(1,8))
        np.testing.assert_array_equal(mask,np.zeros(7))
        np.testing.assert_array_equal(root[:,0],np.ones(7))
        np.testing.assert_array_equal(q,np.zeros((7,27)))

    def test_regular_and_last_window_share_correct_values(self):
        prev={'indices':np.arange(1,8),'root':np.arange(49).reshape(7,7),
              'q':np.arange(189).reshape(7,27)}
        for start in (2,5):
            indices=np.arange(start,start+7)
            root,q,mask=anchor_for(prev,indices)
            common=np.intersect1d(prev['indices'],indices)
            np.testing.assert_array_equal(mask,np.r_[np.ones(len(common)),np.zeros(7-len(common))])
            np.testing.assert_array_equal(root[:len(common)],prev['root'][common-1])
            np.testing.assert_array_equal(q[:len(common)],prev['q'][common-1])
        with self.assertRaises(ValueError):anchor_for(prev,np.arange(10,17))

    def test_windows_cover_all_target_frames(self):
        for n in (8,9,13,64,241):
            starts=window_starts(n)
            self.assertEqual(set(x for s in starts for x in range(s+1,s+8)),set(range(1,n)))

if __name__=='__main__':unittest.main()
