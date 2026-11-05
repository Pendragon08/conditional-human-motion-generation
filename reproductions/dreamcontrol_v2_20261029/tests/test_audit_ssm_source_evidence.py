import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from audit_ssm_source_evidence import audit_ssm_source, audit_ssm_source_bundle


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.a=np.array([[[1.,2.,3.],[2.,3.,4.]],[[1.1,2.,3.],[2.1,3.,4.]]])
        self.b=self.a.copy();self.source=self.root/'source.npz';self.ref=self.root/'ref.npz';self.receipt=self.root/'receipt.json'
        self.record={'sample_id':'id','source_id':'./pose_data/SSM/a'}
        self.align={'motion_id':'id','source_path':self.record['source_id'],'mirrored':False,'source_receipt':str(self.receipt)}
        self.write_inputs()

    def tearDown(self):self.tmp.cleanup()

    def write_inputs(self):
        np.savez(self.source,markers=self.a,labels=np.array(['A','B']),mocap_frame_rate=120.,mocap_time_length=2/120.)
        np.savez(self.ref,marker_data=self.b,marker_labels=np.array([b'A',b'B']),mocap_framerate=60.)
        self.receipt.write_text(json.dumps({'source_group':self.record['source_id'],'smplx_file':str(self.source),
             'reference_file':str(self.ref),'smplx_sha256':sha(self.source),'reference_sha256':sha(self.ref),
             'timeline':{'status':'FRAME_IDENTITY_VERIFIED_BY_MEASURED_MARKERS'}}))

    def audit(self):return audit_ssm_source(self.record,self.align,receipt_path=self.receipt,receipt_sha256=sha(self.receipt))

    def test_unicode_and_bytes_labels_match_but_fps_not_accepted(self):
        result=self.audit();self.assertEqual(result['frames_compared'],2);self.assertTrue(result['header_rates_disagree'])
        self.assertFalse(result['actual_capture_time_accepted']);self.assertFalse(result['training_ready'])

    def test_marker_difference_fails_even_with_rehashed_inputs(self):
        self.b[1,0,1]+=.01;self.write_inputs()
        with self.assertRaisesRegex(ValueError,'do not match'):self.audit()

    def test_missing_or_different_visibility_fails(self):
        self.b[1,0,:]=0;self.write_inputs()
        with self.assertRaisesRegex(ValueError,'visibility'):self.audit()
        self.a[1,:,:]=0;self.b[1,:,:]=0;self.write_inputs()
        with self.assertRaisesRegex(ValueError,'no comparable'):self.audit()

    def test_source_and_receipt_tampering_fails(self):
        self.align['source_path']='different'
        with self.assertRaisesRegex(ValueError,'mapping'):self.audit()
        self.align['source_path']=self.record['source_id'];self.source.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.audit()

    def test_bundle_reports_missing_and_rejects_changed_index(self):
        index=self.root/'align.jsonl';index.write_text(json.dumps(self.align)+'\n')
        manifest=self.root/'evidence.json';manifest.write_text(json.dumps({'alignment_sha256':sha(index),
            'receipts':[{'sample_id':'id','receipt_sha256':sha(self.receipt)}]}))
        missing={'sample_id':'other','source_id':'unrelated'}
        result=audit_ssm_source_bundle([self.record,missing],alignment_path=index,evidence_manifest=manifest)
        self.assertEqual(result['marker_evidence_cases'],1);self.assertEqual(result['samples_without_marker_evidence'],['other'])
        self.assertFalse(result['training_ready'])
        index.write_text(index.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'alignment index'):
            audit_ssm_source_bundle([self.record,missing],alignment_path=index,evidence_manifest=manifest)


if __name__=='__main__':unittest.main()
