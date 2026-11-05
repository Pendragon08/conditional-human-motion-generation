import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_vertical_support_preflight import audit_vertical_support,sampled_vertical_support

class VerticalDiagnosticTests(unittest.TestCase):
    def test_rest_and_ballistic(self):
        t=np.arange(8)*0.05
        p=np.zeros((len(t),3)); p[:,2]=1
        np.testing.assert_allclose(sampled_vertical_support(t,p,-9.81),9.81,atol=1e-12)
        p[:,2]=1-9.81*t*t/2
        np.testing.assert_allclose(sampled_vertical_support(t,p,-9.81),0,atol=1e-12)

    def test_excess_downward_acceleration_and_bad_times(self):
        t=np.arange(8)*.05;p=np.zeros((len(t),3));p[:,2]=-6*t*t
        self.assertTrue(np.all(sampled_vertical_support(t,p,-9.81)<0))
        t[3]=t[2]
        with self.assertRaises(ValueError):sampled_vertical_support(t,p,-9.81)

    def test_missing_and_partial_configuration(self):
        self.assertFalse(audit_vertical_support([{'sample_id':'x'}])['training_ready'])
        with self.assertRaises(ValueError):audit_vertical_support([{'sample_id':'x'}],scene='unused')

if __name__=='__main__':unittest.main()
