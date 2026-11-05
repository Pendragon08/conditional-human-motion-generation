import unittest
import numpy as np
from audit_preview_joint_trajectories import measure


class TrajectoryDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.names=[f'j{i}' for i in range(27)]
        self.limits={name:(-1.,1.) for name in self.names}

    def test_finite_difference_and_limit_excess(self):
        q=np.zeros((3,27));q[:,0]=[0.,.5,1.2]
        report=measure(q,[0.,.1,.2],self.names,self.limits)
        self.assertAlmostEqual(report['max_limit_excess_rad'],.2)
        self.assertEqual(report['max_limit_excess_joint'],'j0')
        self.assertAlmostEqual(report['max_joint_speed_rad_s'],7.)
        self.assertAlmostEqual(report['max_joint_acceleration_rad_s2'],20.)

    def test_invalid_timing_and_joint_names(self):
        q=np.zeros((3,27))
        with self.assertRaises(ValueError):measure(q,[0.,.1,.1],self.names,self.limits)
        with self.assertRaises(ValueError):measure(q,[0.,.1,.2],self.names[:-1]+['j0'],self.limits)


if __name__=='__main__':unittest.main()
