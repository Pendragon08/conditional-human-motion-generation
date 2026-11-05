import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from audit_geometry_caption_binding import audit_geometry_captions


class TestGeometryCaptionBinding(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        p=np.zeros((30,22,3));p[10:20,7,2]=.25;t=np.arange(30)/20
        self.state=self.root/'state.npz';np.savez(self.state,points_z_up=p,sample_times=t)
        h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
        self.row={'sample_id':'x','caption_origin':'machine_geometry_v1','caption_accepted':True,
                  'caption':'The humanoid raises and lowers its left ankle.','feature_sha256':'a'*64}
        entry={'sample_id':'x','caption_origin':'machine_geometry_v1','caption_accepted':False,'training_ready':False,
               'candidate_caption':self.row['caption'],'state_path':'state.npz','state_sha256':h(self.state),
               'feature_sha256':'a'*64,'state_frame_count':30,'feature_frame_count':29,
               'evidence':{'joint_index':7,'window_frames':5,'threshold_m':.12,'center_indices':[2,12,27],
                           'nominal_seconds':[.1,.6,1.35],'mean_z_m':[0.,.25,0.],'rise_m':.25,'fall_m':.25}}
        self.path=self.root/'evidence.jsonl';self.path.write_text(json.dumps(entry)+'\n')
        self.cfg={'evidence_manifest':self.path,'evidence_sha256':h(self.path),'state_root':self.root}
    def tearDown(self):self.tmp.cleanup()
    def test_verifies_bound_geometry(self):
        result=audit_geometry_captions([self.row],**self.cfg)
        self.assertTrue(result['geometry_checks_passed']);self.assertFalse(result['training_ready'])
    def test_rejects_changed_caption(self):
        self.row['caption']='The humanoid kicks a box.'
        with self.assertRaises(ValueError):audit_geometry_captions([self.row],**self.cfg)
    def test_rejects_changed_state(self):
        with self.state.open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):audit_geometry_captions([self.row],**self.cfg)
    def test_rejects_missing_evidence(self):
        with self.assertRaises(ValueError):audit_geometry_captions([self.row])
    def test_pelvis_description_and_direction(self):
        p=np.zeros((30,22,3));p[10:20,0,2]=-.25;t=np.arange(30)/20
        np.savez(self.state,points_z_up=p,sample_times=t)
        entry=json.loads(self.path.read_text());entry.update(caption_origin='machine_geometry_v2',
            candidate_caption='The humanoid lowers and raises its pelvis.',
            state_sha256=hashlib.sha256(self.state.read_bytes()).hexdigest())
        entry['evidence']={'joint_index':0,'direction':-1,'window_frames':5,'threshold_m':.12,
             'center_indices':[2,12,27],'nominal_seconds':[.1,.6,1.35],
             'oriented_mean_z_m':[0.,.25,0.],'rise_m':.25,'fall_m':.25}
        self.path.write_text(json.dumps(entry)+'\n');self.cfg['evidence_sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.row.update(caption_origin='machine_geometry_v2',caption=entry['candidate_caption'])
        self.assertTrue(audit_geometry_captions([self.row],**self.cfg)['geometry_checks_passed'])
        entry['evidence']['direction']=1;self.path.write_text(json.dumps(entry)+'\n')
        self.cfg['evidence_sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):audit_geometry_captions([self.row],**self.cfg)

if __name__=='__main__':unittest.main()
