"""Synthetic structural regressions only; these are not paired training data."""
from pathlib import Path
import sys
import unittest
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'reimplementation'))
from codec262 import G1Codec, quat_matrix
from humanml_pairing import verify_timeline, mirror_fk_witness, mirror_native, native_from_qpos


class PairingTests(unittest.TestCase):
    def fixtures(self):
        rng=np.random.default_rng(42);markers=rng.normal(size=(8,32,3))
        names=np.array(['MARKER_'+str(i) for i in range(32)])
        reference={'marker_labels':names,'marker_data':markers.copy(),'trans':np.zeros((8,3)),
                   'mocap_framerate':np.array(59.9999885559082)}
        target={'labels':names,'markers':markers.copy(),'trans':np.zeros((8,3)),
                'mocap_frame_rate':np.array(120.)}
        return reference,target

    def test_measured_identity_overrides_disagreeing_export_rate(self):
        ref,target=self.fixtures();result=verify_timeline(ref,target)
        self.assertEqual(result['reference_stride'],2)
        self.assertEqual(result['marker_max_abs_m'],0)
        self.assertNotEqual(result['reference_fps'],result['smplx_header_fps'])

    def test_refuse_temporally_shifted_export(self):
        ref,target=self.fixtures();target['markers']=np.roll(target['markers'],1,axis=0)
        with self.assertRaises(ValueError):verify_timeline(ref,target)

    def test_refuse_changed_marker_visibility(self):
        ref,target=self.fixtures();target['markers'][0,0,0]=np.nan
        with self.assertRaises(ValueError):verify_timeline(ref,target)

    def test_zero_triplet_and_nan_are_same_absent_marker(self):
        ref,target=self.fixtures()
        ref['marker_data'][0,0]=0;target['markers'][0,0]=np.nan
        self.assertEqual(verify_timeline(ref,target)['minimum_visible_markers_per_frame'],31)

    def test_refuse_insufficient_marker_evidence(self):
        ref,target=self.fixtures();target['labels']=np.array(['OTHER_'+str(i) for i in range(32)])
        with self.assertRaises(ValueError):verify_timeline(ref,target)

    def test_robot_fk_reflection_and_double_mirror(self):
        codec=G1Codec(BASE/'configs/g1_codec_skeleton.json')
        result=mirror_fk_witness(codec)
        self.assertLess(result['body_position_max_abs_m'],2e-5)
        self.assertLess(result['body_rotation_max_abs'],1e-6)
        q=np.zeros((3,36));q[:,3]=1;q[:,2]=1
        native=native_from_qpos(codec,q,20)
        recovered=mirror_native(codec,mirror_native(codec,native,20),20)
        np.testing.assert_allclose(recovered[:,:3],native[:,:3],atol=1e-12)
        np.testing.assert_allclose(quat_matrix(recovered[:,3:7]),quat_matrix(native[:,3:7]),atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
