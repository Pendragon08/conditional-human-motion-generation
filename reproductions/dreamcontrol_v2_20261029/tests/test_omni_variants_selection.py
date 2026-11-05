"""Check augmented-member selection, caption-key mapping and crop boundary."""
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile
import numpy as np
from resample_antialias_omniretarget import resample_antialias

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')


class VariantsSelectionTest(unittest.TestCase):
    def test_members_are_unique_grouped_and_annotation_keyed(self):
        rows=[json.loads(line) for line in (ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl').read_text().splitlines()]
        variants=[r for r in rows if r['variant']!='original']
        self.assertEqual(len(variants),1514)
        self.assertEqual(len({r['motion_member'] for r in variants}),1514)
        self.assertEqual(len({r['provisional_group'] for r in variants}),6)
        self.assertEqual(len({r['nominal_action_key'] for r in variants}),311)
        with zipfile.ZipFile(ROOT/'data/OmniRetarget_Dataset/robot-object.zip') as z, zipfile.ZipFile(ROOT/'omomo_text_pairing_20261101/omomo_text_anno.zip') as a:
            for row in (variants[0],variants[len(variants)//2],variants[-1]):
                raw=z.read(row['motion_member']);annotation=a.read(row['annotation_member'])
                self.assertEqual(hashlib.sha256(raw).hexdigest(),row['motion_member_sha256'])
                self.assertEqual(hashlib.sha256(annotation).hexdigest(),row['annotation_member_sha256'])
                self.assertEqual(json.loads(annotation)[row['nominal_action_key']],row['caption'])
                with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                    self.assertEqual(float(f['fps']),30.)
                    self.assertEqual(f['qpos'].shape[1],43)

    def test_300_source_frames_give_at_most_199_feature_rows(self):
        q=np.zeros((300,41));q[:,0]=q[:,34]=1.
        out=resample_antialias(q,30.,20.)
        self.assertEqual(len(out)-1,199)
        self.assertLessEqual((len(out)-1)/20.,299/30.)


if __name__=='__main__':unittest.main()
