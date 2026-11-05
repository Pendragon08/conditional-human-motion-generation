import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
HAS_DEPS = all(importlib.util.find_spec(x) is not None for x in ('torch', 'pytorch_kinematics', 'fcntl'))
if HAS_DEPS:
    from batch_fit_audit import select


@unittest.skipUnless(HAS_DEPS, 'requires Linux, torch, and pytorch_kinematics')
class BatchAuditSelectionTest(unittest.TestCase):
    def test_deterministic_source_selection_excludes_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = {n: root/n for n in ('robot-object.zip', 'robot-terrain.zip', 'robot-object-terrain.zip')}
            with zipfile.ZipFile(archives['robot-object.zip'], 'w') as z:
                for i in range(25):
                    z.writestr(f'robot-object/sub1_box_{i:03}_original.npz', b'')
                    z.writestr(f'robot-object/sub1_box_{i:03}_rot_0.npz', b'')
            with zipfile.ZipFile(archives['robot-terrain.zip'], 'w') as z:
                for i in range(25):
                    z.writestr(f'robot-terrain/climb_{i:02}_z_scale_1.0.npz', b'')
                    z.writestr(f'robot-terrain/climb_{i:02}_z_scale_1.1.npz', b'')
            with zipfile.ZipFile(archives['robot-object-terrain.zip'], 'w') as z:
                for i in range(10):
                    z.writestr(f'robot-object-terrain/scene_{i:02}_original.npz', b'')
            a = select(archives, 42); b = select(archives, 42)
            following = select(archives, 99, counts=(5, 5, 0), exclude=set(a))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 50)
        self.assertEqual(len(set(a)), 50)
        self.assertFalse(any('_rot_' in member or '_z_scale_1.1' in member for _, member in a))
        self.assertEqual(len(following), 10)
        self.assertFalse(set(a) & set(following))


if __name__ == '__main__':
    unittest.main()
