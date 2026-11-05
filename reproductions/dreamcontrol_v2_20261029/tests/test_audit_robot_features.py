import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from audit_robot_features import audit


def decode(x): return x[:,:66].reshape(len(x),22,3)


class AuditFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.records=[]; train=[]
        for i, split in enumerate(('train','train','val','test')):
            x=np.full((3,263),i*.1,dtype=np.float32); x[:,-4:]=0
            path=self.root/f'{i}.npz'; np.savez(path,features=x)
            self.records.append({'sample_id':str(i),'group_id':str(i),'split':split,'source_id':f'src{i}',
                                 'caption':'fixture','caption_source':'test','fps':20,'feature_path':path.name,
                                 'feature_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            if split=='train': train.append(x)
        values=np.concatenate(train).astype(np.float64)
        points=decode(values).reshape(len(values),66)
        self.stats={'feature_mean':values.mean(0),'feature_std':np.maximum(values.std(0),.001),
                    'spatial_mean':points.mean(0),'spatial_std':np.maximum(points.std(0),.001)}
    def run_audit(self): return audit(self.root,self.records,self.stats,min_std=.001,decode_points=decode)

    def test_valid_split_stats_and_physical_limit(self):
        report=self.run_audit()
        self.assertTrue(report['technical_checks_passed']); self.assertFalse(report['training_ready'])
        self.assertEqual(report['counts'],{'train':2,'val':1,'test':1})
        self.assertEqual(report['train_frames'],6)

    def test_cross_split_identical_arrays_are_rejected(self):
        path=self.root/'2.npz'
        with np.load(self.root/'0.npz') as data: np.savez(path,features=data['features'])
        self.records[2]['feature_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'cross splits'): self.run_audit()

    def test_stats_derived_from_train_only_and_binary_contact(self):
        self.stats['feature_mean'][0]+=1
        with self.assertRaisesRegex(ValueError,'train-only'): self.run_audit()
        self.stats['feature_mean'][0]-=1
        path=self.root/'3.npz'
        with np.load(path) as data: x=data['features'].copy()
        x[0,-1]=.2; np.savez(path,features=x)
        self.records[3]['feature_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'binary'): self.run_audit()

    def test_group_leakage_and_fps(self):
        self.records[2]['group_id']='0'
        with self.assertRaisesRegex(ValueError,'crosses splits'): self.run_audit()
        self.records[2]['group_id']='2'; self.records[1]['fps']=30
        with self.assertRaisesRegex(ValueError,'20 FPS'): self.run_audit()

    def test_same_source_leakage_and_complex_features(self):
        self.records[2]['source_id']='src0'
        with self.assertRaisesRegex(ValueError,'source crosses splits'): self.run_audit()
        self.records[2]['source_id']='src2'
        path=self.root/'3.npz'; np.savez(path,features=np.ones((3,263),dtype=np.complex64))
        self.records[3]['feature_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'real numeric'): self.run_audit()

    def test_explicit_hierarchical_provenance_blocks_crossing(self):
        for i, record in enumerate(self.records):
            record['provenance_keys'] = {'recording':[f'recording{i}'], 'session':[f'session{i}']}
        self.assertTrue(self.run_audit()['explicit_provenance_groups_disjoint'])
        self.records[2]['provenance_keys']['session'] = ['session0']
        with self.assertRaisesRegex(ValueError, 'provenance relationships cross splits'): self.run_audit()
        del self.records[2]['provenance_keys']
        with self.assertRaisesRegex(ValueError, 'all records or none'): self.run_audit()

    def test_caption_status_remains_declaration_not_acceptance(self):
        self.records[0]['caption_accepted'] = True
        self.records[1]['caption_accepted'] = False
        report=self.run_audit()
        self.assertEqual(report['caption_status_counts'],{'declared_accepted':1,'unaccepted_or_unspecified':3})
        self.assertFalse(report['training_ready'])
        self.records[0]['caption_accepted'] = 'yes'
        with self.assertRaisesRegex(ValueError,'must be boolean'): self.run_audit()


if __name__=='__main__': unittest.main()
