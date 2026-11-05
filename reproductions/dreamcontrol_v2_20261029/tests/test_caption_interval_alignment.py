import unittest
import numpy as np
from caption_interval_alignment import associate_caption


class CaptionAlignmentTests(unittest.TestCase):
    def test_whole_clip_and_cropped_features(self):
        clip=np.arange(6)/20.
        whole=associate_caption(clip_sample_times=clip,feature_sample_times=clip,
                                start_tag='0.0',end_tag='0.0',timed_tag_convention='unknown')
        self.assertEqual(whole['matching_feature_intervals'],list(range(5)))
        self.assertTrue(whole['feature_covers_full_annotation_sample_extent'])
        self.assertFalse(whole['caption_accepted'])
        cut=associate_caption(clip_sample_times=clip,feature_sample_times=clip[1:-1],
                              start_tag='0',end_tag='0',timed_tag_convention='unknown')
        self.assertEqual(cut['matching_feature_intervals'],[0,1,2])
        self.assertFalse(cut['feature_covers_full_annotation_sample_extent'])

    def test_timed_tags_require_explicit_interpretation(self):
        args=dict(clip_sample_times=[10.,10.05,10.1,10.15,10.2],
                  feature_sample_times=[10.,10.05,10.1,10.15,10.2],start_tag='.05',end_tag='.15')
        self.assertEqual(associate_caption(**args,timed_tag_convention='unknown')['status'],'unknown_time_convention')
        result=associate_caption(**args,timed_tag_convention='seconds_relative_to_clip')
        self.assertEqual(result['matching_feature_intervals'],[1,2])
        self.assertFalse(result['caption_accepted'])

    def test_invalid_or_out_of_sample(self):
        args=dict(clip_sample_times=[0.,.05,.1],feature_sample_times=[0.,.05,.1],
                  timed_tag_convention='seconds_relative_to_clip')
        for start,end in (('nan','0'),('0','nan'),('-.1','.1'),('.1','.05')):
            with self.assertRaises(ValueError):associate_caption(**args,start_tag=start,end_tag=end)
        self.assertEqual(associate_caption(**args,start_tag='.05',end_tag='.2')['status'],'out_of_known_sample_range')
        with self.assertRaises(ValueError):
            associate_caption(clip_sample_times=[0.,.05,.1],feature_sample_times=[0.,.05,.15],
                              start_tag='0',end_tag='0',timed_tag_convention='unknown')


if __name__=='__main__':unittest.main()
