"""Check the smooth-min bound and derivative at a sphere-clearance tie."""
import unittest
import numpy as np
from scipy.special import logsumexp,softmax

def smooth(x,tau=.002):return -tau*logsumexp(-np.asarray(x)/tau,axis=-1)

class SmoothSoleTest(unittest.TestCase):
    def test_bias_bound_for_four_clearances(self):
        x=np.random.default_rng(20261030).normal(size=(1000,4))*.02
        exact=x.min(axis=1);value=smooth(x)
        self.assertTrue(np.all(value<=exact+1e-12))
        self.assertTrue(np.all(exact-value<=.002*np.log(4)+1e-12))
    def test_finite_symmetric_derivative_at_tie(self):
        x=np.array([.01,.01,.01,.01]);h=1e-7
        grad=np.array([(smooth(x+np.eye(4)[i]*h)-smooth(x-np.eye(4)[i]*h))/(2*h) for i in range(4)])
        np.testing.assert_allclose(grad,softmax(-x/.002),atol=1e-8,rtol=0)
        np.testing.assert_allclose(grad,np.full(4,.25),atol=1e-8,rtol=0)
if __name__=='__main__':unittest.main()
