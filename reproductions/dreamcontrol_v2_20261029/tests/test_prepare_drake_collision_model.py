import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import trimesh
from prepare_drake_collision_model import prepare, sha256


class PrepareDrakeTests(unittest.TestCase):
    def test_preserves_source_and_converts_verified_triangles(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); mesh=root/'mesh.stl'; trimesh.creation.box().export(mesh)
            urdf=root/'robot.urdf'; urdf.write_text('''<robot name="test"><link name="base">
            <visual><geometry><mesh filename="mesh.stl"/></geometry></visual>
            <collision><geometry><mesh filename="mesh.stl"/></geometry></collision></link></robot>''')
            old_urdf,old_mesh=sha256(urdf),sha256(mesh)
            output=prepare(urdf,root/'generated')
            tree=ET.parse(output); link=tree.getroot().find('link')
            self.assertIsNone(link.find('visual'))
            self.assertEqual(Path(link.find('collision/geometry/mesh').get('filename')).suffix,'.obj')
            self.assertEqual((sha256(urdf),sha256(mesh)),(old_urdf,old_mesh))
            self.assertTrue((root/'generated/manifest.json').exists())
            with self.assertRaises(FileExistsError): prepare(urdf,root/'generated')


if __name__=='__main__': unittest.main()
