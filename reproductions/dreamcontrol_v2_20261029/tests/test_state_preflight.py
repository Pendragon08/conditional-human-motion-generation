import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from train_robot_prior import state_preflight


class StatePreflightTest(unittest.TestCase):
    def test_missing_evidence_cannot_pass(self):
        result = state_preflight(Path('.'), [{'sample_id': 'x'}], {})
        self.assertEqual(result['status'], 'missing_state_evidence')
        self.assertFalse(result['state_technical_checks_passed'])
        self.assertFalse(result['training_ready'])

    def test_rejects_evidence_for_different_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            urdf = root / 'g1.urdf'; urdf.write_text('<robot/>')
            manifest = root / 'state.json'
            manifest.write_text(json.dumps({'records': [{'sample_id': 'x', 'feature_sha256': 'b' * 64}]}))
            cfg = {'state_root': str(root), 'state_manifest': str(manifest), 'state_urdf': str(urdf),
                   'state_urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, 'different features'):
                state_preflight(root, [{'sample_id': 'x', 'feature_sha256': 'a' * 64}], cfg)
            cfg['state_urdf_sha256'] = 'c' * 64
            with self.assertRaisesRegex(ValueError, 'pinned hash'):
                state_preflight(root, [{'sample_id': 'x', 'feature_sha256': 'a' * 64}], cfg)


if __name__ == '__main__':
    unittest.main()
