import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from robot_motion_dataset import RobotMotionDataset, collate_robot_motion


@unittest.skipUnless(HAS_TORCH, 'requires torch and DreamControl data loaders')
class RobotDatasetTest(unittest.TestCase):
    def prepare(self, root):
        records = []
        for index, frames in enumerate([8, 12]):
            path = root/f'clip{index}.npz'
            np.savez_compressed(path, features=np.zeros((frames, 263), dtype=np.float32))
            records.append({'sample_id': f'clip{index}', 'group_id': f'source{index}', 'split': 'train',
                            'caption':'test motion', 'caption_source':'test fixture', 'feature_path': path.name,
                            'feature_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return records

    def load(self, root, records):
        return RobotMotionDataset(root=root, records=records, split='train', feature_mean=np.zeros(263),
                                  feature_std=np.ones(263), spatial_mean=np.zeros((22, 3)),
                                  spatial_std=np.ones((22, 3)))

    def test_collate_padding_and_explicit_zero_hint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); records = self.prepare(root); dataset = self.load(root, records)
            torch.manual_seed(1)
            motion, cond = collate_robot_motion([dataset[0], dataset[1]])
            self.assertEqual(tuple(motion.shape), (2, 263, 1, 12))
            self.assertEqual(cond['y']['lengths'].tolist(), [8, 12])
            self.assertFalse(cond['y']['mask'][0, 0, 0, 8:].any())
            self.assertFalse(cond['y']['hint_mask'][0, 8:].any())
            self.assertTrue(cond['y']['hint_mask'][0].any())
            self.assertTrue(torch.equal(cond['y']['hint'], torch.zeros_like(cond['y']['hint'])))
            self.assertEqual(cond['y']['text'], ['test motion', 'test motion'])

    def test_rejects_cross_split_group_and_bad_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); records = self.prepare(root)
            records[1]['split'] = 'test'; records[1]['group_id'] = records[0]['group_id']
            with self.assertRaisesRegex(ValueError, 'crosses splits'):
                self.load(root, records)
            records[1]['group_id'] = 'other'
            records[0]['feature_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.load(root, records)

    def test_detects_feature_file_change_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); records = self.prepare(root); dataset = self.load(root, records)
            (root/'clip0.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'changed after validation'):
                dataset[0]


if __name__ == '__main__':
    unittest.main()
