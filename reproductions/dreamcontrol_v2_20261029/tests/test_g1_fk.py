import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).parents[1] / "src" / "g1_fk.py"
spec = importlib.util.spec_from_file_location("g1_fk", MODULE)
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)


class ForwardKinematicsTest(unittest.TestCase):
    def test_joint_rotation_and_root_translation(self):
        xml = '''<robot name="fixture"><link name="base"/><link name="elbow"/><link name="tip"/>
        <joint name="turn" type="revolute"><parent link="base"/><child link="elbow"/>
          <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
        <joint name="link" type="fixed"><parent link="elbow"/><child link="tip"/>
          <origin xyz="1 0 0" rpy="0 0 0"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.urdf"
            path.write_text(xml, encoding="utf-8")
            model = fk.URDFFK(path)
            xyz = model.forward(np.array([[0, 0, 0], [3, 2, 0]]),
                                np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
                                np.array([[0], [np.pi / 2]]), ["turn"], links=("tip",))
            np.testing.assert_allclose(xyz[:, 0], [[2, 0, 0], [4, 3, 0]], atol=1e-12)
            pos, rot = model.forward(np.array([[0, 0, 0], [3, 2, 0]]),
                                     np.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
                                     np.array([[0], [np.pi / 2]]), ["turn"], links=("tip",),
                                     return_rotations=True)
            np.testing.assert_allclose(pos, xyz, atol=1e-12)
            np.testing.assert_allclose(rot[1, 0] @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_all_paper_links_exist_in_g1_asset(self):
        asset = Path(__file__).parents[1] / "assets" / "g1_27dof.urdf"
        model = fk.URDFFK(asset)
        self.assertEqual(len(model.actuated), 27)
        self.assertTrue(set(fk.PAPER_22_LINKS) <= model.links)
        with self.assertRaisesRegex(ValueError, "must match"):
            model.forward(np.zeros((1, 3)), np.array([[0, 0, 0, 1]]), np.empty((1, 0)), [])


if __name__ == "__main__":
    unittest.main()
