import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from postopt_generated_bundle import postopt_generated_bundle
import test_robot_prior_postopt as fixtures


class GeneratedBundleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'sample.npz'
        self.output = self.root / 'motion.npz'
        points = np.zeros((2, 4, 22, 3), dtype=np.float32)
        points[1, :, :, 0] = np.arange(4)[:, None]
        raw = json.dumps({'sampling': {'target_frame': 'canonical'}, 'training_or_physical_acceptance': False}).encode()
        np.savez_compressed(self.source, canonical_points=points, metadata_utf8=np.frombuffer(raw, dtype=np.uint8))
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        bridge = fixtures.BridgeTest(); bridge.setUp()
        self.kw = bridge.kw
        self.args = dict(input_path=self.source, input_sha256=self.sha, output_path=self.output,
                         sample_index=1, solver_provenance={'solver': 'synthetic'}, min_free_bytes=0, **self.kw)

    def tearDown(self): self.tmp.cleanup()

    def test_output_provenance_alignment_and_no_overwrite(self):
        result = postopt_generated_bundle(**self.args)
        self.assertFalse(result['training_or_physical_acceptance'])
        self.assertEqual(result['output_sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())
        with np.load(self.output, allow_pickle=False) as data:
            self.assertEqual(data['joint_pos'].shape, (3, 27))
            np.testing.assert_array_equal(data['input_frame_indices'], [1, 2, 3])
            metadata = json.loads(data['metadata_utf8'].tobytes())
            self.assertEqual(metadata['input_sha256'], self.sha)
            self.assertEqual(metadata['sample_index'], 1)
            self.assertFalse(metadata['training_or_physical_acceptance'])
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError): postopt_generated_bundle(**self.args)
        self.assertEqual(before, self.output.read_bytes())

    def test_hash_and_concurrent_input_change_fail_closed(self):
        self.args['input_sha256'] = '0' * 64
        with self.assertRaises(ValueError): postopt_generated_bundle(**self.args)
        self.args['input_sha256'] = self.sha
        solver = self.args['solver']
        def mutate(**kwargs):
            result = solver(**kwargs)
            with self.source.open('ab') as f: f.write(b'changed')
            return result
        self.args['solver'] = mutate
        with self.assertRaises(RuntimeError): postopt_generated_bundle(**self.args)
        self.assertFalse(self.output.exists())

    def test_invalid_index_or_reserve_fails(self):
        self.args['sample_index'] = 2
        with self.assertRaises(ValueError): postopt_generated_bundle(**self.args)
        self.args['sample_index'] = 0
        self.args['min_free_bytes'] = 2**80
        with self.assertRaises(RuntimeError): postopt_generated_bundle(**self.args)
        self.assertFalse(self.output.exists())


if __name__ == '__main__': unittest.main()
