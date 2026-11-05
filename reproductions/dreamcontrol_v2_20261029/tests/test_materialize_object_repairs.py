import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from materialize_object_repairs import main


class MaterializeRepairsTest(unittest.TestCase):
    def test_validated_derived_copy_and_conflict(self):
        q = np.zeros((3, 43)); q[:, 0] = 1.; q[:, 36] = .75
        raw = io.BytesIO(); np.savez_compressed(raw, qpos=q, fps=30)
        payload = raw.getvalue()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'source.zip'; witness = root/'audit.json'; output = root/'out'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr('robot-object-terrain/motion.npz', payload)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            witness.write_text(json.dumps({'archive':'source.zip', 'refused':[], 'repairs':[
                {'member':'robot-object-terrain/motion.npz','source_npz_sha256':hashlib.sha256(payload).hexdigest()}]}))
            with patch.object(sys, 'argv', ['tool', str(source), str(witness), '--output-dir', str(output)]):
                main()
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            with np.load(output/'motion.npz', allow_pickle=False) as data:
                np.testing.assert_allclose(np.linalg.norm(data['qpos'][:, 36:40], axis=1), 1.)
                np.testing.assert_array_equal(data['qpos'][:, :36], q[:, :36])
            self.assertTrue((output/'manifest.json').exists())

    def test_hash_mismatch_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'s.zip'; witness=root/'w.json'; out=root/'out'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr('m.npz', b'not-an-npz')
            witness.write_text(json.dumps({'archive':'s.zip','refused':[],
                                           'repairs':[{'member':'m.npz','source_npz_sha256':'wrong'}]}))
            with patch.object(sys, 'argv', ['tool', str(source), str(witness), '--output-dir', str(out)]):
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    main()
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
