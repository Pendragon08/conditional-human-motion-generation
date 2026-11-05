import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).parents[1] / "src" / "audit_g1_corpus.py"
spec = importlib.util.spec_from_file_location("audit_g1_corpus", MODULE)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


class CorpusAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        motion = np.zeros((6, 36))
        motion[:, 6] = 1.0  # xyzw identity
        motion[:, 20] = 0.2  # waist roll, not waist yaw
        corpus = {"data_dict": {"clip": {"motion": motion, "length": 6, "text": ["walk"]}}}
        with (self.root / "full_train.pkl").open("wb") as f:
            pickle.dump(corpus, f)
        (self.root / "train.txt").write_text("clip\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_reports_locked_waist_and_quaternion(self):
        result = audit_module.audit(self.root, 60, 0.01)
        self.assertEqual(result["clips_exceeding_waist_tolerance"], 1)
        self.assertAlmostEqual(result["waist_roll_pitch_max_abs_rad"], 0.2)
        self.assertEqual(result["quaternion_max_abs_norm_error"], 0)
        self.assertEqual(result["frame_count"], 6)

    def test_overlapping_split_is_rejected(self):
        (self.root / "test.txt").write_text("clip\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sample overlap"):
            audit_module.audit(self.root, 60, 0.01)


if __name__ == "__main__":
    unittest.main()
