import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from summarize_fit_audit import summarize


class FitAuditSummaryTest(unittest.TestCase):
    def write_fixture(self, root):
        samples = [{'archive':'one.zip','member':'one/a.npz'}, {'archive':'two.zip','member':'two/b.npz'}]
        (root/'manifest.json').write_text(json.dumps({'config':{'methods':{'test':[0, 0]}}, 'samples':samples}))
        (root/'COMPLETE.json').write_text(json.dumps({'status':'finished', 'record_count':2}))
        metrics = {x:.01 for x in ('mean_keypoint_error_m','max_keypoint_error_m','mean_foot_error_m',
                                    'max_foot_error_m','mean_wrist_error_m','contact_proxy_velocity_error_mps',
                                    'max_root_translation_m','max_root_rotation_rad')}
        for i,s in enumerate(samples):
            key=hashlib.sha256((s['archive']+'\0'+s['member']).encode()).hexdigest()
            item={**s, 'status':'ok','methods':{'test':metrics}} if i==0 else {**s, 'status':'error','error':'invalid pose'}
            (root/(key+'.json')).write_text(json.dumps(item))
        return samples

    def test_summarizes_success_and_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.write_fixture(root); result=summarize(root)
        self.assertEqual((result['expected'], result['successful'], result['failed']), (2,1,1))
        self.assertEqual(result['methods']['test']['mean_foot_error_m']['mean'], .01)
        self.assertEqual(result['errors'][0]['error'], 'invalid pose')

    def test_refuses_missing_or_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); self.write_fixture(root)
            (root/'COMPLETE.json').write_text(json.dumps({'status':'finished','record_count':1}))
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                summarize(root)


if __name__ == '__main__':
    unittest.main()
