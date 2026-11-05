import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from length_bucket_sampler import ExactLengthBatchSampler


class ExactLengthBatchSamplerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lengths = [3, 5, 3, 5, 3, 5, 3, 5, 7]
        records = []
        for i, n in enumerate(self.lengths):
            p = Path(self.tmp.name)/f'{i}.npz'
            np.savez(p, features=np.zeros((n, 263), dtype=np.float32))
            stat = p.stat()
            records.append({'_path':p, '_file_size':stat.st_size, '_file_mtime_ns':stat.st_mtime_ns})
        self.data = SimpleNamespace(records=records, min_frames=2, max_frames=199)

    def test_batches_are_same_length_and_cover_every_sample(self):
        sampler = ExactLengthBatchSampler(self.data, 3, seed=7)
        batches = list(sampler)
        self.assertEqual(len(sampler), len(batches))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(9)))
        self.assertTrue(all(len({self.lengths[i] for i in b}) == 1 for b in batches))
        self.assertEqual(batches, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(batches, list(sampler))

    def test_drop_last_and_changed_input(self):
        sampler = ExactLengthBatchSampler(self.data, 3, drop_last=True, shuffle=False)
        self.assertEqual(list(sampler), [[0,2,4], [1,3,5]])
        with self.data.records[0]['_path'].open('ab') as f:
            f.write(b'changed')
        with self.assertRaises(RuntimeError):
            ExactLengthBatchSampler(self.data, 3)

    def test_invalid_arguments(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                ExactLengthBatchSampler(self.data, value)


if __name__ == '__main__':
    unittest.main()
