import hashlib,tempfile,unittest
from pathlib import Path
import torch
from robot_prior_checkpoint import load_robot_prior_checkpoint


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.body=torch.nn.Linear(3,2);self.hint_mask_embedding=torch.nn.Linear(22,2,bias=False)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.stats=self.root/'stats.npz';self.stats.write_bytes(b'synthetic statistics')
        self.manifest='a'*64;self.path=self.root/'checkpoint.pt'
        self.source=Model();self.target=Model()
        self.payload={'version':1,'model':self.source.state_dict(),'optimizer':{},'rng':{},
                      'provenance':{'dataset_manifest_sha256':self.manifest,'stats_sha256':self.sha(self.stats),'model_init_sha256':'c'*64},
                      'cursor':{'step':3,'epoch':0,'next_batch':0},'config':{'seed':7}}
        self.save()
    def tearDown(self):self.tmp.cleanup()
    @staticmethod
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def save(self):torch.save(self.payload,self.path)
    def load(self,**kwargs):
        return load_robot_prior_checkpoint(self.target,self.path,checkpoint_sha256=kwargs.get('checkpoint_sha256',self.sha(self.path)),
                                           stats_path=self.stats,dataset_manifest_sha256=kwargs.get('manifest',self.manifest))
    def test_loads_bound_nonzero_step_checkpoint(self):
        result=self.load();self.assertEqual(result['step'],3)
        self.assertFalse(result['training_or_physical_acceptance'])
        for key,value in self.source.state_dict().items():self.assertTrue(torch.equal(value,self.target.state_dict()[key]))
    def test_wrong_pins_or_untrained_base_rejected(self):
        with self.assertRaisesRegex(ValueError,'checkpoint hash mismatch'):self.load(checkpoint_sha256='b'*64)
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):self.load(manifest='b'*64)
        self.payload['cursor']['step']=0;self.save()
        with self.assertRaisesRegex(ValueError,'completed training step'):self.load()
        self.payload=self.source.state_dict();self.save()
        with self.assertRaisesRegex(ValueError,'not a base model'):self.load()
    def test_bad_model_rejected_before_mutation(self):
        before={k:v.clone() for k,v in self.target.state_dict().items()}
        self.payload['model']['body.weight']=torch.full_like(self.payload['model']['body.weight'],float('nan'));self.save()
        with self.assertRaisesRegex(ValueError,'nonfinite'):self.load()
        for key,value in before.items():self.assertTrue(torch.equal(value,self.target.state_dict()[key]))
        self.payload['model']['body.weight']=torch.zeros((1,1));self.save()
        with self.assertRaisesRegex(ValueError,'shape or dtype'):self.load()
        for key,value in before.items():self.assertTrue(torch.equal(value,self.target.state_dict()[key]))


if __name__=='__main__':unittest.main()
