import tempfile
import unittest
from pathlib import Path
import numpy as np
from audit_declared_joint_velocity import audit_velocity, read_velocity_limits


class VelocityAuditTest(unittest.TestCase):
    def test_nonuniform_times_and_named_limits(self):
        limits={'j0':2., 'j1':4.}
        result=audit_velocity([[0,0],[.2,.4],[.4,.6]],[0,.1,.3],['j0','j1'],limits)
        self.assertAlmostEqual(result['maximum_ratio'],1.)
        self.assertEqual(result['joint_interval_pairs_above_limit'],0)
        result=audit_velocity([[0,0],[.3,.4],[.4,.6]],[0,.1,.3],['j0','j1'],limits)
        self.assertAlmostEqual(result['maximum_ratio'],1.5)
        self.assertEqual(result['worst_joint'],'j0')
        self.assertEqual(result['worst_interval'],0)
        self.assertEqual(result['intervals_with_any_joint_above_limit'],1)
        self.assertFalse(result['training_ready'])

    def test_missing_limits_and_bad_times_fail_closed(self):
        self.assertEqual(audit_velocity([[0],[1]],[0,.1],['j'],{'j':None})['status'],
                         'missing_declared_limits')
        with self.assertRaises(ValueError): audit_velocity([[0],[1]],[0,0],['j'],{'j':1})
        with self.assertRaises(ValueError): audit_velocity([[0],[1]],[0,.1],['j'],{})
        with self.assertRaises(ValueError): audit_velocity([[0],[np.nan]],[0,.1],['j'],{'j':1})

    def test_urdf_declarations_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.urdf'
            p.write_text('<robot><joint name="a" type="revolute"><limit velocity="2"/></joint>'
                         '<joint name="b" type="revolute"><limit/></joint></robot>')
            self.assertEqual(read_velocity_limits(p),{'a':2.,'b':None})
            p.write_text('<robot><joint name="a" type="revolute"><limit velocity="0"/></joint></robot>')
            with self.assertRaises(ValueError): read_velocity_limits(p)


if __name__=='__main__': unittest.main()
