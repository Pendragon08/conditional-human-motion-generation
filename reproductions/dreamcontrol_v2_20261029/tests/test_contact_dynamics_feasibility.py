import unittest
import numpy as np
from contact_dynamics_feasibility import check_force_balance


def solve(accel,bias,jac,act=None,lo=(),hi=(),friction=(.5,)):
    d=len(accel)
    if act is None:act=np.zeros((d,0))
    return check_force_balance(np.eye(d),accel,bias,act,np.asarray(jac,dtype=float).reshape(-1,3,d),
                               torque_lower=lo,torque_upper=hi,friction=friction,tolerance=1e-8)


class ForceBalanceTest(unittest.TestCase):
    def test_stationary_gravity_requires_support(self):
        jac=[[[1,0],[0,0],[0,1]]]
        r=solve([0,0],[0,9.81],jac)
        self.assertEqual(r['status'],'feasible')
        self.assertAlmostEqual(r['contact_forces'][0][2],9.81)
        self.assertFalse(r['training_ready'])
        r=solve([0,0],[0,9.81],[],friction=())
        self.assertEqual(r['status'],'infeasible_no_forces')

    def test_friction_and_motor_bounds(self):
        jac=[[[1,0],[0,0],[0,1]]]
        self.assertEqual(solve([4,0],[0,10],jac)['status'],'feasible')
        self.assertEqual(solve([6,0],[0,10],jac)['status'],'infeasible_inner_friction_pyramid')
        self.assertEqual(solve([0],[2],[],act=np.array([[1.]]),lo=[-1],hi=[1],friction=())['status'],
                         'infeasible_force_balance')
        self.assertEqual(solve([0],[.5],[],act=np.array([[1.]]),lo=[-1],hi=[1],friction=())['status'],'feasible')

    def test_inner_pyramid_failure_does_not_mean_circular_cone_failure(self):
        # f=(3.5,3.5,10) is inside the circular mu=.5 cone but outside the diamond.
        jac=[np.eye(3)]
        r=solve([3.5,3.5,0],[0,0,10],jac)
        self.assertEqual(r['status'],'infeasible_inner_friction_pyramid')
        self.assertLess(np.hypot(3.5,3.5),5.)

    def test_two_contacts_and_actuators_reconstruct_generalized_forces(self):
        m=np.diag([2.,3.,4.]); a=np.array([.1,-.2,.3])
        act=np.array([[1.,0.],[0.,1.],[.3,.4]])
        jac=np.array([[[1.,0.,.2],[0.,1.,.1],[0.,0.,1.]],
                      [[.5,.1,0.],[.1,.5,.2],[0.,.1,1.]]])
        known=np.array([[.2,-.1,1.],[0.,.1,1.]])
        rhs=act@np.array([.2,-.3])+sum(j.T@f for j,f in zip(jac,known))
        bias=rhs-m@a
        r=check_force_balance(m,a,bias,act,jac,torque_lower=[-2,-2],torque_upper=[2,2],
                              friction=[.5,.5],tolerance=1e-8)
        self.assertEqual(r['status'],'feasible')
        forces=np.array(r['contact_forces']);torques=np.array(r['torques'])
        recomputed=act@torques+sum(j.T@f for j,f in zip(jac,forces))
        self.assertTrue(np.allclose(recomputed,rhs,rtol=0,atol=1e-8))
        self.assertTrue(all(f[2]>=-1e-8 and abs(f[0])+abs(f[1])<=.5*f[2]+1e-8 for f in forces))
        self.assertTrue(np.all(np.abs(torques)<=2.+1e-8))

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):solve([0,0],[0,np.nan],[[[1,0],[0,0],[0,1]]])
        with self.assertRaises(ValueError):
            check_force_balance([[0]],[0],[0],np.zeros((1,0)),np.zeros((0,3,1)),
                                torque_lower=[],torque_upper=[],friction=[],tolerance=1e-8)


if __name__=='__main__':unittest.main()
