import unittest
import numpy as np
from omni_variant_caption_time import temporal_summary


class CaptionTimeTest(unittest.TestCase):
    def test_cropped_moving_object(self):
        q=np.zeros((330,43));q[:,40]=np.arange(330)/30.
        result=temporal_summary(q,30.)
        self.assertEqual(result['crop_frames'],300)
        self.assertEqual(result['nominal_feature_rows'],199)
        self.assertEqual(result['target_last_source_frame_position'],298.5)
        self.assertAlmostEqual(result['target_to_full_end_center_m'],(329-298.5)/30.)
        self.assertEqual(result['source_tail_frames_after_crop'],30)
        self.assertFalse(result['full_caption_valid_for_variant_or_window'])

    def test_short_static_object_and_invalid_inputs(self):
        q=np.zeros((61,43))
        result=temporal_summary(q,30.)
        self.assertEqual(result['nominal_feature_rows'],40)
        self.assertEqual(result['target_to_later_max_center_m'],0.)
        with self.assertRaises(ValueError):temporal_summary(q,29.)
        with self.assertRaises(ValueError):temporal_summary(np.zeros((3,42)),30.)
        q[0,40]=np.nan
        with self.assertRaises(ValueError):temporal_summary(q,30.)


if __name__=='__main__':unittest.main()
