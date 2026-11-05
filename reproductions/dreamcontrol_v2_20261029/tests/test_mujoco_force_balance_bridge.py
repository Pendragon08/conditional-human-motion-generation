import unittest
import numpy as np
import mujoco
from mujoco_force_balance_bridge import evaluate


class MuJoCoBridgeTest(unittest.TestCase):
    def setUp(self):
        self.model=mujoco.MjModel.from_xml_string('''<mujoco>
          <option gravity="0 0 -9.81"/>
          <worldbody><body name="box" pos="0 0 .1"><freejoint/>
          <geom type="box" size=".1 .1 .1" mass="2"/></body></worldbody></mujoco>''')
        self.state=self.model.qpos0.copy();self.zero=np.zeros(self.model.nv)
        self.contact={'body':'box','point_world':[0,0,0],'basis_world':np.eye(3).tolist(),'friction':.5}

    def check(self,contacts):
        return evaluate(self.model,self.state,self.zero,self.zero,actuated_joint_names=[],
                        torque_lower=[],torque_upper=[],contacts=contacts,tolerance=1e-7)

    def test_static_box_and_no_support(self):
        r=self.check([self.contact])
        self.assertEqual(r['status'],'feasible')
        self.assertAlmostEqual(r['contact_forces'][0][2],19.62,places=7)
        self.assertFalse(r['training_ready'])
        self.assertEqual(self.check([])['status'],'infeasible_no_forces')

    def test_rotated_contact_basis_preserves_world_normal(self):
        self.contact['basis_world']=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
        r=self.check([self.contact]);self.assertEqual(r['status'],'feasible')
        self.assertAlmostEqual(r['contact_forces'][0][2],19.62,places=7)

    def test_hinge_effort_limit_uses_named_dof(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 -9.81"/>
          <worldbody><body name="arm"><joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 1 0 0" size=".05" mass="2"/>
          </body></worldbody></mujoco>''')
        def run(limit):
            return evaluate(model,model.qpos0,np.zeros(model.nv),np.zeros(model.nv),
                actuated_joint_names=['elbow'],torque_lower=[-limit],torque_upper=[limit],
                contacts=[],tolerance=1e-7)
        result=run(20.);self.assertEqual(result['status'],'feasible')
        self.assertAlmostEqual(abs(result['torques'][0]),9.81,places=7)
        self.assertEqual(run(1.)['status'],'infeasible_force_balance')

    def test_invalid_frame_and_quaternion_rejected(self):
        self.contact['basis_world']=[[1,0,0],[0,1,0],[0,0,-1]]
        with self.assertRaisesRegex(ValueError,'basis'):self.check([self.contact])
        self.state[3:7]*=2
        with self.assertRaisesRegex(ValueError,'quaternion'):self.check([])


if __name__=='__main__':unittest.main()
