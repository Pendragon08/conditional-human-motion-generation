import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from combine_fit_audits import combine


def make(root, name):
    root.mkdir()
    cfg={'methods':{'m':[0,0]},'steps':120,'learning_rate':.015,'urdf29_sha256':'a','urdf27_sha256':'b',
         'source_archives_sha256':{'a':'c'},'dependency_sha256':{'d':'e'},'torch_version':'v','numpy_version':'v'}
    item={'archive':'a.zip','member':name}
    (root/'manifest.json').write_text(json.dumps({'config':cfg,'samples':[item]}))
    (root/'COMPLETE.json').write_text(json.dumps({'status':'finished','record_count':1}))
    keys=('mean_keypoint_error_m','max_keypoint_error_m','mean_foot_error_m','max_foot_error_m',
          'mean_wrist_error_m','contact_proxy_velocity_error_mps','max_root_translation_m','max_root_rotation_rad')
    k=hashlib.sha256(('a.zip\0'+name).encode()).hexdigest()
    (root/(k+'.json')).write_text(json.dumps({**item,'status':'ok','methods':{'m':{k:.01 for k in keys}}}))


class CombineTest(unittest.TestCase):
    def test_compatible_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=p/'a';b=p/'b';c=p/'c';make(a,'one');make(b,'two');make(c,'one')
            self.assertEqual(combine([a,b])['successful'],2)
            with self.assertRaisesRegex(ValueError,'duplicate'):
                combine([a,c])

    def test_rejects_incompatible_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=p/'a';b=p/'b';make(a,'one');make(b,'two')
            m=json.loads((b/'manifest.json').read_text());m['config']['steps']=10
            (b/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'differ'):
                combine([a,b])


if __name__=='__main__': unittest.main()
