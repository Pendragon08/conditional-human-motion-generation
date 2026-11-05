import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).parents[1] / 'src' / 'nymeria_manifest_preflight.py'
spec = importlib.util.spec_from_file_location('nymeria_manifest_preflight', MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ManifestPreflightTest(unittest.TestCase):
    def test_counts_only_selected_groups_without_disclosing_url(self):
        private_link = 'https://example.invalid/?credential=secret'
        item = {'file_size_bytes': 123, 'sha1sum': 'abc', 'download_url': private_link}
        manifest = {'sequences': {'seq1': {'body_motion': item, 'recording_head': item},
                                  'seq2': {'body_motion': item}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'urls.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            result = mod.inspect(path)
        self.assertEqual(result['sequence_count'], 2)
        self.assertEqual(result['compressed_download_bytes'], 246)
        self.assertEqual(result['groups']['body_motion']['count'], 2)
        self.assertEqual(result['groups']['narration_atomic_action_csv']['missing'], 2)
        self.assertNotIn('secret', json.dumps(result))


if __name__ == '__main__':
    unittest.main()
