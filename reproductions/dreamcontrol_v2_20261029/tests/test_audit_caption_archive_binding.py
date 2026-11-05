import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_caption_archive_binding import annotation_range, audit_caption_archive

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj))

class CaptionArchiveTests(unittest.TestCase):
    def test_annotation_modes(self):
        self.assertEqual(annotation_range(['0.0','0.0'],[0.,3.]),((0.,3.),'whole_clip_convention'))
        self.assertEqual(annotation_range(['0.5','2.0'],[0.,3.]),((.5,2.),'timed_segment'))
        for tags in (['nan','0'],['1','1'],['-1','2'],['0','4']):
            with self.assertRaises(ValueError):annotation_range(tags,[0.,3.])

    def test_missing_configuration(self):
        self.assertEqual(audit_caption_archive([])['status'],'missing_archive_binding')
        with self.assertRaises(ValueError):audit_caption_archive([],archive='only-one')

    def test_end_to_end_and_changed_tags(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);archive=root/'texts.zip';state=root/'state.npz'
            raw=b'person walks#person/NOUN walks/VERB#0.0#0.0\n'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('texts/000001.txt',raw)
            np.savez(state,sample_times=np.array([.0,.05,.1]))
            feature='f'*64
            receipt={'sample_id':'s','caption':'person walks','candidate_sha256':feature,
                'caption_archive_sha256':sha(archive),'caption_member_sha256':hashlib.sha256(raw).hexdigest(),
                'selected_line':0,'tags':['0.0','0.0'],'source_range_s':[0.,.1],'sample_range_s':[0.,.1]}
            receipt_path=root/'receipt.json';write(receipt_path,receipt)
            evidence=root/'evidence.json';write(evidence,{'receipts':[{'sample_id':'s','receipt_path':'receipt.json','receipt_sha256':sha(receipt_path)}]})
            state_manifest=root/'states.json';write(state_manifest,{'records':[{'sample_id':'s','state_path':'state.npz','state_sha256':sha(state),'feature_sha256':feature}]})
            records=[{'sample_id':'s','caption':'person walks','caption_source':'HumanML texts.zip: texts/000001.txt',
                      'caption_accepted':True,'feature_sha256':feature}]
            kwargs={'archive':archive,'evidence_root':root,'evidence_manifest':evidence,'state_root':root,'state_manifest':state_manifest}
            result=audit_caption_archive(records,**kwargs)
            self.assertEqual(result['cases'][0]['feature_intervals'],2)
            self.assertFalse(result['training_ready'])
            receipt['tags']=['0.0','1.0'];write(receipt_path,receipt)
            write(evidence,{'receipts':[{'sample_id':'s','receipt_path':'receipt.json','receipt_sha256':sha(receipt_path)}]})
            with self.assertRaisesRegex(ValueError,'raw tags mismatch'):audit_caption_archive(records,**kwargs)

if __name__=='__main__':unittest.main()
