import hashlib,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from generate_robot_prior import generate_robot_prior
from test_robot_prior_sampling import SmallConditionalModel,diffusion


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.manifest=self.root/'manifest.json';self.manifest.write_text('{"records": []}')
        self.stats=self.root/'stats.npz'
        np.savez(self.stats,feature_mean=np.zeros(263),feature_std=np.ones(263),spatial_mean=np.zeros(66),spatial_std=np.ones(66))
        self.model=SmallConditionalModel().eval();self.diff=diffusion();self.ckpt=self.root/'model.pt'
        payload={'version':1,'model':self.model.state_dict(),'optimizer':{},'rng':{},'config':{'test':True},
                 'cursor':{'step':1,'epoch':0,'next_batch':0},
                 'provenance':{'dataset_manifest_sha256':self.sha(self.manifest),'stats_sha256':self.sha(self.stats),'model_init_sha256':'c'*64}}
        torch.save(payload,self.ckpt)
        self.out=self.root/'output.npz';self.targets=torch.zeros((1,6,22,3));self.mask=torch.zeros((1,6,22),dtype=torch.bool)
        self.mask[0,3,1]=True
    def tearDown(self):self.tmp.cleanup()
    @staticmethod
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def run_generation(self):
        return generate_robot_prior(self.model,self.diff,checkpoint_path=self.ckpt,checkpoint_sha256=self.sha(self.ckpt),
            manifest_path=self.manifest,stats_path=self.stats,output_path=self.out,texts=['a person walks'],
            targets=self.targets,target_mask=self.mask,cfg_scale=1.,spatial_step_size=.1,spatial_steps=1,seed=12,min_free_bytes=0)
    def test_bundle_and_no_overwrite(self):
        result=self.run_generation();self.assertEqual(result['output_sha256'],self.sha(self.out))
        with np.load(self.out,allow_pickle=False) as f:
            self.assertEqual(f['normalized_features'].shape,(1,263,1,6))
            self.assertEqual(f['canonical_points'].shape,(1,6,22,3))
            self.assertTrue(f['target_mask'][0,3,1]);meta=json.loads(f['metadata_utf8'].tobytes().decode())
        self.assertEqual(meta['checkpoint']['step'],1);self.assertFalse(meta['training_or_physical_acceptance'])
        digest=self.sha(self.out)
        with self.assertRaises(FileExistsError):self.run_generation()
        self.assertEqual(self.sha(self.out),digest)
    def test_bad_manifest_and_stats_fail_before_output(self):
        self.manifest.write_text('{"records": [1]}')
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):self.run_generation()
        self.assertFalse(self.out.exists())
        self.manifest.write_text('{"records": []}')
        np.savez(self.stats,feature_mean=np.zeros(263),feature_std=np.zeros(263),spatial_mean=np.zeros(66),spatial_std=np.ones(66))
        with self.assertRaisesRegex(ValueError,'standard deviations'):self.run_generation()
        self.assertFalse(self.out.exists())


if __name__=='__main__':unittest.main()
