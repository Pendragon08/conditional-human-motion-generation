import tempfile
import unittest
from pathlib import Path
import numpy as np
from foot_geometry_audit import compare, foot_spheres, sphere_centers
from g1_fk import URDFFK


class FootGeometryTests(unittest.TestCase):
    def test_rotated_collision_centers(self):
        xml='''<robot name="test"><link name="base"/>
        <link name="left_ankle_roll_link"><collision><origin xyz="0 0 -0.1"/><geometry><sphere radius="0.005"/></geometry></collision></link>
        <link name="right_ankle_roll_link"><collision><origin xyz="0 0 -0.1"/><geometry><sphere radius="0.005"/></geometry></collision></link>
        <joint name="left" type="fixed"><parent link="base"/><child link="left_ankle_roll_link"/></joint>
        <joint name="right" type="fixed"><parent link="base"/><child link="right_ankle_roll_link"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'robot.urdf'; path.write_text(xml)
            offsets,radii=foot_spheres(path)
            q=np.array([[0,np.sqrt(.5),0,np.sqrt(.5)]])
            centers=sphere_centers(URDFFK(path),np.array([[0,0,1.]]),q,np.empty((1,0)),[],offsets)
            np.testing.assert_allclose(centers[0,:,0,:],[[-.1,0,1],[-.1,0,1]],atol=1e-12)
            np.testing.assert_allclose(radii,.005)

    def test_penetration_and_stance_speed(self):
        source=np.zeros((3,2,1,3)); source[...,2]=.005
        target=source.copy(); target[...,2]=.002; target[:, :, 0, 0]=np.arange(3)[:,None]*.01
        result=compare(source,target,np.full((2,1),.005),fps=20,plane_z=0,height_tolerance=.01,max_vertical_speed=.1)
        self.assertAlmostEqual(result['target_max_penetration_m'],.003)
        self.assertAlmostEqual(result['target_contact_proxy_mean_tangent_speed_mps'],.2)
        self.assertEqual(result['source_contact_proxy_samples'],4)

    def test_no_contact_and_invalid_plane(self):
        centers=np.ones((2,2,1,3)); radii=np.full((2,1),.005)
        result=compare(centers,centers,radii,fps=20,plane_z=0,height_tolerance=.01,max_vertical_speed=.1)
        self.assertIsNone(result['target_contact_proxy_mean_tangent_speed_mps'])
        with self.assertRaises(ValueError): compare(centers,centers,radii,fps=0,plane_z=0,height_tolerance=.01,max_vertical_speed=.1)


if __name__=='__main__': unittest.main()
