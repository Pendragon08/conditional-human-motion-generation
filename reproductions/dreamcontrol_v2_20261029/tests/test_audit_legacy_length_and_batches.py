import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_legacy_length_and_batches import legacy_policy

SOURCE='''
class Text2MotionDatasetV2:
    def __init__(self):
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
        if len(motion) < min_motion_len or len(motion) >= 200:
            pass
'''
class LegacyPolicyTests(unittest.TestCase):
    def test_pinned_structure_and_limits(self):
        self.assertEqual(legacy_policy(SOURCE,'t2m'),(40,200))
        self.assertEqual(legacy_policy(SOURCE,'kit'),(24,200))
    def test_changed_or_missing_structure_fails_closed(self):
        with self.assertRaises(ValueError):legacy_policy(SOURCE.replace('len(motion) >= 200','len(other) >= 200'),'t2m')
        with self.assertRaises(ValueError):legacy_policy('class Other: pass','t2m')
if __name__=='__main__':unittest.main()
