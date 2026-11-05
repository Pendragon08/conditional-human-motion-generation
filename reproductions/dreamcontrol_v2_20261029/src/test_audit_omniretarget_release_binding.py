import hashlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import audit_omniretarget_release_binding as mod

class TestReleaseBinding(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        raw=io.BytesIO();np.savez(raw,qpos=np.zeros((10,43)),fps=np.array(30.));payload=raw.getvalue()
        self.archive=self.root/'robot-object.zip'
        with zipfile.ZipFile(self.archive,'w') as z:z.writestr('robot-object/sample.npz',payload)
        self.card=self.root/'README.md';self.card.write_text('---\nlicense: mit\n---\nrobot-object\n')
        self.feature=self.root/'feature.npz';np.savez(self.feature,features=np.zeros((6,263)))
        self.state=self.root/'state.npz';np.savez(self.state,sample_times=np.arange(7)/20.)
        self.row={'sample_id':'sample','source_member':'robot-object/sample.npz','source_sha256':hashlib.sha256(payload).hexdigest(),
                  'feature_path':'feature.npz','feature_sha256':mod.sha(self.feature)}
        self.states=self.root/'states.json';self.states.write_text(json.dumps({'records':[{'sample_id':'sample',
            'state_path':'state.npz','state_sha256':mod.sha(self.state),'feature_sha256':mod.sha(self.feature)}]}))
        self.cfg={'root':self.root,'state_manifest':self.states,'archive':self.archive,'card':self.card}
        self.patches=[patch.object(mod,'ARCHIVE_SHA',mod.sha(self.archive)),patch.object(mod,'CARD_SHA',mod.sha(self.card))]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()
    def test_real_bytes_and_time(self):
        r=mod.audit_release_binding([self.row],**self.cfg)
        self.assertTrue(r['release_bytes_verified']);self.assertFalse(r['training_ready'])
    def test_missing_evidence_fails_closed(self):
        self.assertFalse(mod.audit_release_binding([self.row])['release_bytes_verified'])
    def test_rejects_changed_source_hash(self):
        self.row['source_sha256']='0'*64
        with self.assertRaises(ValueError):mod.audit_release_binding([self.row],**self.cfg)
    def test_rejects_timing_beyond_source(self):
        np.savez(self.state,sample_times=np.arange(7)/10.)
        value=json.loads(self.states.read_text());value['records'][0]['state_sha256']=mod.sha(self.state)
        self.states.write_text(json.dumps(value))
        with self.assertRaises(ValueError):mod.audit_release_binding([self.row],**self.cfg)
    def test_rejects_changed_card(self):
        self.card.write_text('changed')
        with self.assertRaises(ValueError):mod.audit_release_binding([self.row],**self.cfg)
if __name__=='__main__':unittest.main()
