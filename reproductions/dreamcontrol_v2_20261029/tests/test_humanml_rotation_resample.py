import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from humanml_time_mapping import aligned_grid
from humanml_rotation_resample import apply_grid


class ResampleTests(unittest.TestCase):
    def test_real_time_30_to_20_matches_independent_slerp(self):
        fps=59.9999885559082;n=121
        t=np.arange(n)/fps
        xyz=np.stack([t,2*t,-t],axis=1)
        # Cross the axis-angle branch cut and use a second rotating joint.
        pose=np.zeros((n,6));pose[:,2]=3.0+0.8*t;pose[:,3]=0.4*t
        grid=aligned_grid(n,fps,0,-1)
        pos,out=apply_grid(xyz,pose,grid)
        src=np.array(grid['source_indices']); src_t=(src-src[0])/fps
        target=np.array([w['time'] for w in grid['weights']])
        np.testing.assert_allclose(pos[:,0],target+t[src[0]],atol=1e-12)
        for j in range(2):
            expected=Slerp(src_t,Rotation.from_rotvec(pose[src,j*3:j*3+3]))(target)
            actual=Rotation.from_rotvec(out[:,j*3:j*3+3])
            self.assertLess(np.max((expected.inv()*actual).magnitude()),1e-7)

    def test_constant_rotations_and_bounds(self):
        grid=aligned_grid(121,120.0,0,-1)
        pos,rot=apply_grid(np.ones((121,3)),np.zeros((121,6)),grid)
        np.testing.assert_allclose(pos,1);np.testing.assert_allclose(rot,0)
        bad={'weights':[{'left_frame':0,'right_frame':121,'alpha':0.5}]}
        with self.assertRaises(ValueError):apply_grid(np.ones((121,3)),np.zeros((121,6)),bad)


if __name__=='__main__':unittest.main()
