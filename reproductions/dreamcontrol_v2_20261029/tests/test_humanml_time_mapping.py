import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from humanml_time_mapping import aligned_grid


class TimeMappingTests(unittest.TestCase):
    def test_near_60_float_preserves_reference_times_and_actual_20hz(self):
        out = aligned_grid(301, 59.9999885559082, 0, -1)
        self.assertEqual(out['reference_stride'], 2)
        self.assertGreater(out['actual_reference_sample_fps'], 29.9)
        times = [w['time'] for w in out['weights']]
        self.assertTrue(all(abs((b-a)-0.05) < 1e-10 for a,b in zip(times,times[1:])))
        self.assertLessEqual(times[-1], out['source_duration_s'] + 1e-10)
        for item in out['weights']:
            reconstructed = ((1-item['alpha'])*item['left_frame'] + item['alpha']*item['right_frame'] - out['source_indices'][0]) / 59.9999885559082
            self.assertAlmostEqual(reconstructed,item['time'],places=9)

    def test_trim_crop_and_legacy_negative_stop(self):
        out = aligned_grid(180,120.0,2,-1,prefix_trim=3)
        self.assertEqual(out['source_indices'], list(range(30,174,6)))
        self.assertEqual(out['weights'][0]['left_frame'],30)
        self.assertAlmostEqual(out['weights'][-1]['time'],1.15)

    def test_out_of_bounds_rejected(self):
        with self.assertRaises(ValueError): aligned_grid(30,120.0,0,100)


if __name__=='__main__': unittest.main()
