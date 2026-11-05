import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audit_grab import audit


class GrabAuditTest(unittest.TestCase):
    def test_reads_metadata_without_unpickling_object_fields(self):
        payload = io.BytesIO()
        np.savez(payload, sbj_id='s2', framerate=120., n_frames=240,
                 motion_intent='use', obj_name='mug', body={'params': 1},
                 object={'params': 2}, contact={'data': 3})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'subject.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('s2/mug_use.npz', payload.getvalue())
            result = audit(path)
        self.assertEqual(result['metadata_valid'], 1)
        self.assertEqual(result['subject_counts'], {'s2': 1})
        self.assertEqual(result['fps_counts'], {'120.0': 1})
        self.assertEqual(result['total_frames'], 240)

    def test_invalid_missing_metadata_is_reported(self):
        payload = io.BytesIO()
        np.savez(payload, sbj_id='s2')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'subject.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('s2/bad.npz', payload.getvalue())
            result = audit(path)
        self.assertEqual(result['metadata_valid'], 0)
        self.assertEqual(len(result['errors']), 1)


if __name__ == '__main__':
    unittest.main()
