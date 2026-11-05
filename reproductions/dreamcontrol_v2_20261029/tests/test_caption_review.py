import hashlib
import tempfile
import unittest
from pathlib import Path

from render_caption_review import checked_file, sample_indices


class CaptionReviewTest(unittest.TestCase):
    def test_indices_cover_first_last_and_increase(self):
        self.assertEqual(sample_indices(11, 6).tolist(), [0, 2, 4, 6, 8, 10])
        with self.assertRaises(ValueError):
            sample_indices(5, 6)

    def test_state_identity_and_root_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / 'state.npz'
            path.write_bytes(b'test state')
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(checked_file(root, path.name, checksum), path)
            with self.assertRaises(ValueError):
                checked_file(root, path.name, '0' * 64)
            with self.assertRaises(ValueError):
                checked_file(root, str(path), checksum)


if __name__ == '__main__':
    unittest.main()
