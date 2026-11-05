import unittest
import numpy as np
from pyroki_ssm_rawfps_batch_gpu import window_starts


class WindowCoverageTest(unittest.TestCase):
    def test_clip_coverage_and_overlap(self):
        for length in (8,9,12,233,303,399):
            starts = window_starts(length)
            windows = [set(range(s+1,s+8)) for s in starts]
            self.assertEqual(set.union(*windows), set(range(1,length)))
            self.assertEqual(starts[0],0)
            self.assertEqual(starts[-1],length-8)
            for a,b in zip(windows,windows[1:]):
                self.assertTrue(a & b)

    def test_nonzero_source_offset(self):
        source = np.arange(26,425)
        starts = window_starts(len(source))
        self.assertEqual(int(source[starts[0]+1]),27)
        self.assertEqual(int(source[starts[-1]+7]),424)

    def test_too_short_fails(self):
        with self.assertRaises(ValueError): window_starts(7)


if __name__ == '__main__': unittest.main()
