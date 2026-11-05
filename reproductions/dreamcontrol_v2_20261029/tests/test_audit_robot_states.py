import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from audit_robot_states import audit_robot_states


class StateAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.urdf = self.root / 'g1.urdf'
        self.names = ['j{}'.format(i) for i in range(27)]
        self.urdf.write_text('<robot name="g1">' + ''.join(
            '<joint name="{}" type="revolute"><limit lower="-1" upper="1"/></joint>'.format(n)
            for n in self.names) + '</robot>')
        self.feature = self.root / 'feature.npz'
        np.savez(self.feature, features=np.zeros((2, 263), dtype=np.float32))
        self.state = self.root / 'state.npz'
        self.q = np.zeros((3, 27))
        self.times = np.array([.0, .05, .10])
        self.write_state()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_state(self):
        base = np.zeros((3, 7)); base[:, 0] = 1.
        np.savez(self.state, root_wxyz_xyz=base, joint_pos=self.q,
                 sample_times=self.times, joint_names=np.array(self.names))

    def run_audit(self, canonicalize=lambda x: x, decode=lambda x: np.zeros((len(x), 22, 3)),
                  reencode=lambda x, foot_velocity_squared_threshold: np.zeros((len(x) - 1, 263))):
        records = [{'sample_id': 'x', 'feature_path': self.feature.name, 'feature_sha256': self.sha(self.feature),
                    'state_path': self.state.name, 'state_sha256': self.sha(self.state),
                    'foot_velocity_squared_threshold': .002}]
        return audit_robot_states(records, feature_root=self.root, state_root=self.root, urdf_path=self.urdf,
                                  tolerance_rad=1e-6, fk_decode_tolerance_m=1e-4, reencoding_tolerance=1e-5,
                                  fk=lambda pos, rot, q, names: np.zeros((len(q), 22, 3)),
                                  decode=decode, canonicalize=canonicalize, reencode=reencode)

    def test_pass_and_joint_failure_are_separate_from_training(self):
        valid = self.run_audit()
        self.assertTrue(valid['state_technical_checks_passed'])
        self.assertEqual(valid['cases'][0]['encoding_variant'], 'custom_callable_unidentified')
        self.assertFalse(valid['training_ready'])
        self.q[1, 1] = 1.1
        self.write_state()
        failed = self.run_audit()
        self.assertFalse(failed['state_technical_checks_passed'])
        self.assertEqual(failed['limit_failed_samples'], ['x'])

    def test_direct_29_dof_source_layout_and_waist_limit(self):
        self.names = ['j{}'.format(i) for i in range(29)]
        self.urdf.write_text('<robot name="g1">' + ''.join(
            '<joint name="{}" type="revolute"><limit lower="-1" upper="1"/></joint>'.format(n)
            for n in self.names) + '</robot>')
        packed = np.zeros((3, 43)); packed[:, 0] = 1.; packed[:, 36] = 1.
        np.savez(self.state, qpos29_source_layout_20hz=packed,
                 sample_times=self.times, joint_names=np.array(self.names))
        self.assertTrue(self.run_audit()['state_technical_checks_passed'])
        packed[1, 21] = 1.1
        np.savez(self.state, qpos29_source_layout_20hz=packed,
                 sample_times=self.times, joint_names=np.array(self.names))
        self.assertFalse(self.run_audit()['state_technical_checks_passed'])

    def test_reject_direct_29_dof_state_against_27_dof_urdf(self):
        packed = np.zeros((3, 43)); packed[:, 0] = 1.; packed[:, 36] = 1.
        np.savez(self.state, qpos29_source_layout_20hz=packed,
                 sample_times=self.times, joint_names=np.array(self.names))
        with self.assertRaisesRegex(ValueError, '29-DoF'):
            self.run_audit()

    def test_reject_bad_time_and_nonfinite_state(self):
        self.times[1] = .04
        self.write_state()
        with self.assertRaises(ValueError):
            self.run_audit()
        self.times[1] = .05
        self.q[1, 1] = np.nan
        self.write_state()
        with self.assertRaises(ValueError):
            self.run_audit()

    def test_reject_nonfinite_canonical_points(self):
        with self.assertRaises(ValueError):
            self.run_audit(canonicalize=lambda x: np.full_like(x, np.nan))

    def test_reject_mismatched_features(self):
        with self.assertRaisesRegex(ValueError, 'state/feature FK consistency'):
            self.run_audit(decode=lambda x: np.ones((len(x), 22, 3)))

    def test_reject_mismatched_rotation_channel_with_same_decoded_points(self):
        def wrong_rotation(points, foot_velocity_squared_threshold):
            values = np.zeros((len(points) - 1, 263))
            values[0, 67] = .01
            return values
        with self.assertRaisesRegex(ValueError, 'full feature re-encoding'):
            self.run_audit(reencode=wrong_rotation)

    def test_mixed_named_encoders_verify_each_record(self):
        corrected = self.root / 'signfixed.npz'
        data = np.zeros((2, 263), dtype=np.float32); data[:, 0] = .125
        np.savez(corrected, features=data)
        def legacy(points, foot_velocity_squared_threshold):
            return np.zeros((len(points)-1, 263), dtype=np.float32)
        def signfixed(points, foot_velocity_squared_threshold):
            result = legacy(points, foot_velocity_squared_threshold); result[:, 0] = .125
            return result
        shared = {'state_path': self.state.name, 'state_sha256': self.sha(self.state),
                  'foot_velocity_squared_threshold': .002}
        records = [dict(shared, sample_id='old', feature_path=self.feature.name, feature_sha256=self.sha(self.feature)),
                   dict(shared, sample_id='new', feature_path=corrected.name, feature_sha256=self.sha(corrected),
                        encoding_variant='sign_continuous_v1')]
        kwargs = dict(feature_root=self.root, state_root=self.root, urdf_path=self.urdf,
                      tolerance_rad=1e-6, fk_decode_tolerance_m=1e-4, reencoding_tolerance=1e-5,
                      fk=lambda pos, rot, q, names: np.zeros((len(q), 22, 3)),
                      decode=lambda x: np.zeros((len(x), 22, 3)), canonicalize=lambda x: x)
        result = audit_robot_states(records, reencode={'legacy':legacy,'sign_continuous_v1':signfixed}, **kwargs)
        self.assertTrue(result['state_technical_checks_passed'])
        self.assertEqual([r['encoding_variant'] for r in result['cases']], ['legacy','sign_continuous_v1'])
        self.assertFalse(result['training_ready'])
        records[1]['encoding_variant']='legacy'
        with self.assertRaisesRegex(ValueError, 'full feature re-encoding'):
            audit_robot_states(records, reencode={'legacy':legacy,'sign_continuous_v1':signfixed}, **kwargs)
        records[1]['encoding_variant']='unknown'
        with self.assertRaisesRegex(ValueError, 'unsupported or unavailable'):
            audit_robot_states(records, reencode={'legacy':legacy,'sign_continuous_v1':signfixed}, **kwargs)
        records[1]['encoding_variant']='sign_continuous_v1'
        with self.assertRaisesRegex(ValueError, 'explicit encoder map'):
            audit_robot_states(records, reencode=signfixed, **kwargs)


if __name__ == '__main__':
    unittest.main()
