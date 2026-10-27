"""Guard against FPS floor-stride, conflicting exports and unsupported identity."""
import unittest
from gate_phygile_sample_clock import clock_check


class SampleClockTests(unittest.TestCase):
    def timeline(self,ref=120.00004577636719,target=120,stride=6):
        return {'status':'FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS',
                'reference_fps':ref,'smplx_header_fps':target,
                'reference_stride':stride,'nominal_output_fps':20}

    def test_small_numeric_header_roundoff_is_accepted(self):
        self.assertTrue(clock_check(self.timeline())['eligible'])

    def test_same_headers_do_not_hide_floor_stride_speed_change(self):
        result=clock_check(self.timeline(59.9999886,59.9999886,2))
        self.assertFalse(result['eligible'])
        self.assertNotIn('conflicting_export_fps_headers',result['reasons'])
        self.assertIn('reference_stride_clock_differs_from_feature_clock',result['reasons'])

    def test_twofold_export_conflict_is_rejected(self):
        self.assertFalse(clock_check(self.timeline(59.9999886,120,2))['eligible'])

    def test_matching_fps_cannot_replace_frame_identity(self):
        x=self.timeline();x['status']='FITTED_POSES_SIMILAR'
        self.assertFalse(clock_check(x)['eligible'])

    def test_invalid_values_cannot_pass(self):
        for key,value in [('reference_fps',float('nan')),('reference_stride',0),
                          ('reference_stride',True),('nominal_output_fps',-20)]:
            x=self.timeline();x[key]=value
            self.assertFalse(clock_check(x)['eligible'])


if __name__=='__main__':unittest.main()
