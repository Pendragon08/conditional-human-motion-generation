import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_caption_receipts import audit_caption_receipts


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.receipt={'sample_id':'a','caption':'bend knees','candidate_sha256':'a'*64,
            'sample_range_s':[.1,.9],'source_range_s':[0.,1.],
            'manual_semantic_assessment':{'semantic_accepted_for_exported_samples':True,
                 'reviewer':'test','reason':'synthetic','status':'test'},'training_ready':False}
        self.path=self.root/'review.json';self.path.write_text(json.dumps(self.receipt))
        self.check=self.root/'audit.json';self.check.write_text(json.dumps({'receipt_sha256':digest(self.path),'training_ready':False}))
        self.row={'sample_id':'a','caption':'bend knees','caption_accepted':True,'feature_sha256':'a'*64,
                  'caption_review_sha256':digest(self.path)}
        self.bundle=self.root/'bundle.json';self.save_bundle()

    def tearDown(self):self.tmp.cleanup()

    def save_bundle(self):
        self.bundle.write_text(json.dumps({'receipts':[{'sample_id':'a','receipt_path':'review.json','receipt_sha256':digest(self.path),
            'independent_audit_path':'audit.json','independent_audit_sha256':digest(self.check)}]}))

    def run_check(self):
        return audit_caption_receipts([self.row],evidence_root=self.root,evidence_manifest=self.bundle)

    def test_valid_receipt_never_authorizes_training(self):
        r=self.run_check();self.assertEqual(r['verified_semantic_receipts'],1);self.assertFalse(r['training_ready'])

    def test_accepted_without_evidence_fails(self):
        with self.assertRaises(ValueError):audit_caption_receipts([self.row])
        r=audit_caption_receipts([{'sample_id':'b','caption_accepted':False}]);self.assertEqual(r['unaccepted_captions'],1)

    def test_tampered_caption_feature_or_receipt_fails(self):
        self.row['caption']='different'
        with self.assertRaises(ValueError):self.run_check()
        self.row['caption']='bend knees';self.row['feature_sha256']='b'*64
        with self.assertRaises(ValueError):self.run_check()
        self.row['feature_sha256']='a'*64;self.path.write_text('{}')
        with self.assertRaises(ValueError):self.run_check()

    def test_outside_root_and_invalid_time_rejected(self):
        b=json.loads(self.bundle.read_text());b['receipts'][0]['receipt_path']='../review.json';self.bundle.write_text(json.dumps(b))
        with self.assertRaises((ValueError,FileNotFoundError)):self.run_check()
        self.receipt['sample_range_s']=[-.1,.9];self.path.write_text(json.dumps(self.receipt));self.row['caption_review_sha256']=digest(self.path)
        self.check.write_text(json.dumps({'receipt_sha256':digest(self.path),'training_ready':False}));self.save_bundle()
        with self.assertRaises(ValueError):self.run_check()


if __name__=='__main__':unittest.main()
