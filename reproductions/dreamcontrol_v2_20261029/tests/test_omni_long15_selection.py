"""Check the fixed long-track selection and 10-second sampling boundary."""
import json
import unittest
from pathlib import Path
import numpy as np
from resample_antialias_omniretarget import resample_antialias

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')


class LongTrackSelectionTest(unittest.TestCase):
    def test_new_long_sources_do_not_repeat_prior_batches(self):
        rows=[json.loads(line) for line in (ROOT/'omomo_object_caption_probe_20261101/frames_summary.jsonl').read_text().splitlines()]
        selected=[row for row in rows if row['nominal_last_sample_s']>9.95]
        self.assertEqual(len(selected),15)
        members={row['motion_member'] for row in selected}
        self.assertEqual(len(members),15)
        for folder in ('omni_gpu_batch24_20261101','omni_gpu_remaining_20261101'):
            old=set(json.loads((ROOT/folder/'selection.json').read_text())['members'])
            self.assertFalse(members & old)
        pairs={json.loads(line)['motion_member'] for line in (ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl').read_text().splitlines()}
        self.assertTrue(members <= pairs)
        self.assertTrue(all(row['frames']>300 for row in selected))

    def test_crop_never_exceeds_sample_support(self):
        q=np.zeros((300,41),dtype=float)
        q[:,0]=1.;q[:,34]=1.
        out=resample_antialias(q,30.,20.)
        self.assertEqual(len(out),200)
        self.assertLessEqual((len(out)-1)/20.,299/30.)


if __name__=='__main__':
    unittest.main()
