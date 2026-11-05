import unittest
import numpy as np

from contact_interval_view import contact_interval_view


class ContactIntervalViewTest(unittest.TestCase):
    def test_short_event_is_distinct_from_first_sample(self):
        # Same frame indices as the observed S4 label interval; synthetic labels.
        labels = np.zeros((12, 1), dtype=bool)
        labels[5:8] = True
        result = contact_interval_view(labels, np.ones_like(labels), [0, 6, 12], source_frame_start=180, declared_source_fps=120)
        np.testing.assert_array_equal(result['first_sample_positive'][:, 0], [False, True])
        np.testing.assert_array_equal(result['any_known_positive'][:, 0], [True, True])
        np.testing.assert_array_equal(result['source_frame_begin_inclusive'], [180, 186])
        np.testing.assert_array_equal(result['source_frame_end_exclusive'], [186, 192])
        np.testing.assert_allclose(result['positive_frame_fraction_lower'][:, 0], [1/6, 2/6])
        self.assertEqual(result['declared_source_fps'], 120)

    def test_unknown_is_not_negative_and_positive_still_has_unknown(self):
        label = np.array([[False, False], [True, False], [False, False]], dtype=bool)
        valid = np.array([[False, True], [True, True], [True, False]], dtype=bool)
        r = contact_interval_view(label, valid, [0, 3])
        np.testing.assert_array_equal(r['any_known_positive'][0], [True, False])
        np.testing.assert_array_equal(r['any_known_positive_known'][0], [True, False])
        np.testing.assert_array_equal(r['fully_observed'][0], [False, False])
        np.testing.assert_allclose(r['positive_frame_fraction_lower'][0], [1/3, 0])
        np.testing.assert_allclose(r['positive_frame_fraction_upper'][0], [2/3, 1/3])
        self.assertIsNone(r['declared_source_fps'])

    def test_tail_is_explicit_and_inputs_unchanged(self):
        x = np.array([[False], [True], [False], [True], [False]], dtype=bool)
        valid = np.ones_like(x); original = x.copy()
        r = contact_interval_view(x, valid, [0, 3, 5])
        np.testing.assert_array_equal(r['source_frame_end_exclusive'], [3, 5])
        np.testing.assert_array_equal(x, original)
        with self.assertRaises(ValueError): contact_interval_view(x, valid, [0, 3])

    def test_invalid_edges_rates_and_labels_fail(self):
        x = np.zeros((4, 1), dtype=bool)
        cases = [lambda: contact_interval_view(x.astype(int), x, [0, 4]),
                 lambda: contact_interval_view(x, x, [0, 3, 2, 4]),
                 lambda: contact_interval_view(x, x, np.array([0, 3, 2, 4], dtype=np.uint64)),
                 lambda: contact_interval_view(x, x, [0, 4], declared_source_fps=float('nan')),
                 lambda: contact_interval_view(x, x, [0, 4], source_frame_start=np.iinfo(np.int64).max)]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError): case()


if __name__ == '__main__':
    unittest.main()
