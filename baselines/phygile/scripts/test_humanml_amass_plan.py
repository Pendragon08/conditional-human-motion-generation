"""Temporal boundary regression tests; no real-motion or physics performance claims."""
import unittest
from plan_humanml_amass import original_frame_indices, parse_caption, source_parts


class TemporalAlignmentTests(unittest.TestCase):
    def test_trim_before_index_then_caption(self):
        row = {'source_path': './pose_data/MPI_HDM05/a/b_poses.npy',
               'start_frame': 10, 'end_frame': 100}
        caption = parse_caption('fixture#tokens#0.5#1.0')
        indices = original_frame_indices(row, 120, 1500, caption)
        self.assertEqual(indices, list(range(480, 540, 6)))

    def test_python_minus_one_excludes_last_sample(self):
        row = {'source_path': './pose_data/KIT/a/b_poses.npy',
               'start_frame': 0, 'end_frame': -1}
        self.assertEqual(original_frame_indices(row, 60, 30), list(range(0, 27, 3)))

    def test_fractional_fps_uses_original_stride(self):
        row = {'source_path': './pose_data/KIT/a/b_poses.npy',
               'start_frame': 0, 'end_frame': 5}
        self.assertEqual(original_frame_indices(row, 59.94, 30), [0, 2, 4, 6, 8])

    def test_whole_and_subframe_rounding(self):
        self.assertEqual(parse_caption('fixture#tokens#0#0')['kind'], 'whole')
        self.assertEqual(parse_caption('fixture#tokens#0.099#0.151')['start_20'], 1)

    def test_reject_invalid_caption_tags(self):
        for tags in ('nan#0', '0#inf', '2#1', '-1#2', '0.001#0.002'):
            with self.subTest(tags=tags), self.assertRaises(ValueError):
                parse_caption('fixture#tokens#' + tags)

    def test_refuse_silent_clamping(self):
        row = {'source_path': './pose_data/KIT/a/b_poses.npy',
               'start_frame': 0, 'end_frame': 10}
        with self.assertRaises(ValueError):
            original_frame_indices(row, 60, 60, parse_caption('fixture#tokens#0#1'))

    def test_humanact12_requires_separate_adapter(self):
        row = {'source_path': './pose_data/humanact12/a/b.npy',
               'start_frame': 0, 'end_frame': -1}
        with self.assertRaises(ValueError):
            original_frame_indices(row, 60, 60)

    def test_forbid_parent_paths(self):
        with self.assertRaises(ValueError):
            source_parts('./pose_data/KIT/../secret.npy')


if __name__ == '__main__':
    unittest.main(verbosity=2)
