import unittest
import numpy as np
from ground_calibration import fit_ground_surface


def fit(p,**kwargs):
    args=dict(labels=['a','b','c'],observation_kind='fixed_ground_surface_points',source_to_robot=np.eye(4),
              source_up=[0,0,1],stationary_tolerance_m=.001,plane_tolerance_m=.001,min_normal_alignment=.99)
    args.update(kwargs)
    return fit_ground_surface(p,**args)


class GroundTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([[[0.,0.,.3],[1.,0.,.3],[0.,1.,.3]],[[0.,0.,.3],[1.,0.,.3],[0.,1.,.3]]])

    def test_valid_surface_and_rigid_transform(self):
        t=np.eye(4);t[:3,3]=[1,2,.4]
        result=fit(self.p,source_to_robot=t)
        np.testing.assert_allclose(result['normal_robot'],[0,0,1],atol=1e-12)
        self.assertAlmostEqual(result['offset_robot'],-.7)
        self.assertFalse(result['training_ready'])

    def test_body_markers_cannot_calibrate_floor(self):
        with self.assertRaisesRegex(ValueError,'body markers'):fit(self.p,observation_kind='body_surface_markers')

    def test_reject_collinear_and_nonfinite(self):
        line=np.array([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]]*2)
        with self.assertRaises(ValueError):fit(line)
        p=self.p.copy();p[0,0,0]=np.nan
        with self.assertRaises(ValueError):fit(p)

    def test_reject_motion_bad_transform_and_vertical_plane(self):
        p=self.p.copy();p[1,0,2]+=.01
        with self.assertRaises(ValueError):fit(p)
        t=np.eye(4);t[0,0]=-1
        with self.assertRaises(ValueError):fit(self.p,source_to_robot=t)
        p=np.array([[[0.,0.,0.],[0.,1.,0.],[0.,0.,1.]]]*2)
        with self.assertRaises(ValueError):fit(p)

    def test_reject_insufficient_time_and_nonplanar_points(self):
        with self.assertRaises(ValueError):fit(self.p[:1])
        p=np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,.01]]]*2)
        with self.assertRaises(ValueError):fit(p,labels=['a','b','c','d'])


if __name__=='__main__':unittest.main()
