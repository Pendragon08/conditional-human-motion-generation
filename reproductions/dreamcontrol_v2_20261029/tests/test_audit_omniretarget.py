import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit_omniretarget import audit_archive


def npz(**items):
    output = io.BytesIO()
    np.savez_compressed(output, **items)
    return output.getvalue()


class ArchiveAuditTest(unittest.TestCase):
    def test_valid_and_invalid_members_are_reported(self):
        valid = np.zeros((5, 36))
        valid[:, 0] = 1
        bad_quaternion = np.zeros((5, 36))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motions.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('data/ok.npz', npz(qpos=valid, fps=np.array(20)))
                archive.writestr('data/bad.npz', npz(qpos=bad_quaternion, fps=np.array(20)))
                archive.writestr('data/ignored.txt', 'hello')
            result = audit_archive(path)
        self.assertEqual(result['members'], 2)
        self.assertEqual(result['valid'], 1)
        self.assertEqual(result['fps_counts'], {'20.0': 1})
        self.assertEqual(result['qpos_width_counts'], {'36': 1})
        self.assertEqual(len(result['errors']), 1)
        self.assertIn('zero base quaternion', result['errors'][0]['error'])
        self.assertFalse(result['joint_order_verified'])

    def test_rejects_invalid_object_orientation(self):
        q = np.zeros((2, 43))
        q[:, 0] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motions.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('motion.npz', npz(qpos=q, fps=np.array(30)))
            result = audit_archive(path)
        self.assertEqual(result['valid'], 0)
        self.assertIn('zero object quaternion', result['errors'][0]['error'])


if __name__ == '__main__':
    unittest.main()
