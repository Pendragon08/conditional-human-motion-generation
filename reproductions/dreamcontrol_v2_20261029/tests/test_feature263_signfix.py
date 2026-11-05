import unittest
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from feature263 import canonicalize_z_up,encode
from feature263_signfix import align_signs,encode_signfixed

SOURCE=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/omni_gpu_batch24_20261101')
class SignFixTest(unittest.TestCase):
    def test_sign_alignment_preserves_orientations(self):
        q=np.array([[1.,0,0,0],[-1.,0,0,0],[1.,0,0,0]])
        np.testing.assert_array_equal(align_signs(q),np.tile([1.,0,0,0],(3,1)))
        with self.assertRaises(ValueError):align_signs(q*.5)

    def test_quarantined_clip_recovery_and_channel_scope(self):
        with np.load(SOURCE/'sub8_largebox_006_20hz.npz',allow_pickle=False) as z:points=z['points_z_up']
        raw=encode(points,foot_velocity_squared_threshold=.002)
        fixed=encode_signfixed(points,foot_velocity_squared_threshold=.002)
        np.testing.assert_array_equal(raw[:,1:],fixed[:,1:])
        self.assertEqual(np.flatnonzero(raw[:,0]!=fixed[:,0]).tolist(),[94])
        with torch.no_grad():
            original=recover_from_ric(torch.from_numpy(raw),22).numpy()
            corrected=recover_from_ric(torch.from_numpy(fixed),22).numpy()
        truth=canonicalize_z_up(points)[:-1]
        self.assertGreater(float(np.linalg.norm(original-truth,axis=-1).max()),.05)
        self.assertLess(float(np.linalg.norm(corrected-truth,axis=-1).max()),1e-4)

if __name__=='__main__':unittest.main()
