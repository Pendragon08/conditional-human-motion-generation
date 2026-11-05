import hashlib,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_manual_caption_binding import audit_manual_captions
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.write_text(json.dumps(o))

class ManualCaptionTests(unittest.TestCase):
    def test_empty_and_missing(self):
        self.assertEqual(audit_manual_captions([])['status'],'no_manual_captions')
        with self.assertRaises(ValueError):audit_manual_captions([{'sample_id':'s'}])
    def test_binding_and_time_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);state=p/'state.npz';np.savez(state,sample_times=np.array([1.,1.05,1.1]))
            annot=p/'annotation.json';audit=p/'audit.json';ev=p/'evidence.json';st=p/'states.json';f='f'*64
            annotation={'sample_id':'s','caption':'arm down','feature_sha256':f,'caption_origin':'manual visual annotation; not provided by source',
                'reviewer':'reviewer','semantic_supported_for_exported_window':True,'training_ready':False,
                'source_frames':[120,126,132],'feature_rows':2,'nominal_time_s':[1.,1.1]}
            save(annot,annotation);save(audit,{'annotation_sha256':sha(annot),'training_ready':False})
            def evidence():save(ev,{'receipts':[{'sample_id':'s','annotation_path':'annotation.json','annotation_sha256':sha(annot),
                                                'audit_path':'audit.json','audit_sha256':sha(audit),'declared_source_fps':120}]})
            evidence();save(st,{'records':[{'sample_id':'s','state_path':'state.npz','state_sha256':sha(state),'feature_sha256':f}]})
            row={'sample_id':'s','caption':'arm down','caption_accepted':True,'caption_origin':'manual_visual',
                 'caption_review_sha256':sha(annot),'feature_sha256':f}
            kwargs={'evidence_root':p,'evidence_manifest':ev,'state_root':p,'state_manifest':st}
            self.assertEqual(audit_manual_captions([row],**kwargs)['cases'][0]['feature_intervals'],2)
            annotation['source_frames']=[120,127,132];save(annot,annotation)
            save(audit,{'annotation_sha256':sha(annot),'training_ready':False});evidence();row['caption_review_sha256']=sha(annot)
            with self.assertRaisesRegex(ValueError,'timeline differs'):audit_manual_captions([row],**kwargs)

    def test_interpolated_positions_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);state=p/'state.npz'; times=np.array([1.,1.05,1.1]);np.savez(state,sample_times=times,raw_source_frames=np.arange(59,68))
            annot=p/'annotation.json';audit=p/'audit.json';ev=p/'evidence.json';st=p/'states.json';f='a'*64
            rate=59.9999885559082
            annotation={'sample_id':'s','caption':'arm forward','feature_sha256':f,
                'caption_origin':'manual visual annotation; not provided by source','reviewer':'reviewer',
                'semantic_supported_for_exported_window':True,'training_ready':False,
                'timeline_mapping':'linear_source_frame_positions','sample_source_frame_positions':(times*rate).tolist(),
                'source_frame_bounds':[59,67], 'feature_rows':2,'nominal_time_s':[1.,1.1]}
            def refresh():
                save(annot,annotation);save(audit,{'annotation_sha256':sha(annot),'training_ready':False})
                save(ev,{'receipts':[{'sample_id':'s','annotation_path':'annotation.json','annotation_sha256':sha(annot),
                    'audit_path':'audit.json','audit_sha256':sha(audit),'declared_source_fps':rate}]})
                return {'sample_id':'s','caption':'arm forward','caption_accepted':True,'caption_origin':'manual_visual',
                        'caption_review_sha256':sha(annot),'feature_sha256':f}
            save(st,{'records':[{'sample_id':'s','state_path':'state.npz','state_sha256':sha(state),'feature_sha256':f}]})
            kwargs={'evidence_root':p,'evidence_manifest':ev,'state_root':p,'state_manifest':st}
            row=refresh()
            self.assertEqual(audit_manual_captions([row],**kwargs)['cases'][0]['timeline_mapping'],'linear_source_frame_positions')
            annotation['source_frame_bounds']=[60,67];row=refresh()
            with self.assertRaisesRegex(ValueError,'bounds'):audit_manual_captions([row],**kwargs)
            annotation['source_frame_bounds']=[59,67];annotation['sample_source_frame_positions'][1]+=.1;row=refresh()
            with self.assertRaisesRegex(ValueError,'timeline differs'):audit_manual_captions([row],**kwargs)
            annotation['sample_source_frame_positions'][1]-=.1
            annotation['source_frame_bounds']=[58,68];row=refresh()
            with self.assertRaisesRegex(ValueError,'hashed state support'):audit_manual_captions([row],**kwargs)
if __name__=='__main__':unittest.main()
