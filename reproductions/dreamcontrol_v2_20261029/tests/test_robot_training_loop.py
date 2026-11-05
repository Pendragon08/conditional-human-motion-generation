import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from robot_training_loop import run_training


class FixtureDataset:
    def __init__(self, root):
        self.records=[]; self.min_frames=2; self.max_frames=199
        for i, n in enumerate((4,5,4,5,4)):
            path = Path(root)/f'{i}.npz'
            np.savez(path,features=np.ones((n,263),dtype=np.float32))
            stat=path.stat()
            self.records.append({'_path':path,'_file_size':stat.st_size,'_file_mtime_ns':stat.st_mtime_ns})
    def __getitem__(self,i):
        with np.load(self.records[i]['_path'],allow_pickle=False) as f:
            x=torch.tensor(f['features'])
        n=len(x)
        # Simulate stochastic conditions, making RNG restoration observable.
        return {'sample_id':str(i),'text':'test','motion':x + torch.rand(())*.1,
                'hint':torch.zeros(n,66),'hint_mask':torch.zeros(n,22,dtype=torch.bool)}


class DummyDiffusion:
    model_mean_type=SimpleNamespace(name='START_X')
    posterior_log_variance_clipped=[0.]*10
    def q_sample(self,x,t,noise): return x + noise*.1


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.5))
    def forward(self,x,t,y): return x*self.weight


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.dataset=FixtureDataset(self.root)
        self.provenance={'dataset_manifest_sha256':'a'*64,'stats_sha256':'b'*64,'model_init_sha256':'c'*64}
        self.config={'batch_size':2,'seed':5,'guidance_steps':0,'guidance_scale':.5}
    def run_model(self, model, optimizer, directory, max_steps, resume=None, log=None):
        return run_training(model,DummyDiffusion(),optimizer,self.dataset,run_dir=directory,
                            provenance=self.provenance,config=self.config,max_steps=max_steps,checkpoint_every=2,
                            batch_size=2,seed=5,guidance_steps=0,guidance_scale=.5,
                            device=torch.device('cpu'),resume_from=resume,on_step=log)

    def test_resume_matches_uninterrupted_including_random_data(self):
        full=self.root/'full'; full.mkdir()
        resumed=self.root/'resumed'; resumed.mkdir()
        a=DummyModel(); oa=torch.optim.AdamW(a.parameters(),lr=.01)
        b=DummyModel(); ob=torch.optim.AdamW(b.parameters(),lr=.01)
        losses_full=[]; losses_split=[]
        self.run_model(a,oa,full,6,log=lambda m,c:losses_full.append(m['loss']))
        self.run_model(b,ob,resumed,3,log=lambda m,c:losses_split.append(m['loss']))
        c=DummyModel(); oc=torch.optim.AdamW(c.parameters(),lr=.01)
        result=self.run_model(c,oc,resumed,6,resume=resumed/'step_000000003.pt',log=lambda m,c:losses_split.append(m['loss']))
        self.assertEqual(losses_full,losses_split)
        self.assertTrue(torch.equal(a.weight,c.weight))
        self.assertEqual(result['cursor']['step'],6)
        self.assertFalse((resumed/'training.lock').exists())

    def test_concurrent_lock_and_nonempty_run_are_rejected(self):
        directory=self.root/'locked'; directory.mkdir()
        (directory/'training.lock').write_text('other job')
        model=DummyModel(); optim=torch.optim.AdamW(model.parameters())
        with self.assertRaises(FileExistsError): self.run_model(model,optim,directory,1)
        self.assertEqual((directory/'training.lock').read_text(),'other job')
        other=self.root/'nonempty'; other.mkdir(); (other/'keep').write_text('keep')
        with self.assertRaisesRegex(ValueError,'empty'): self.run_model(model,optim,other,1)
        self.assertEqual((other/'keep').read_text(),'keep')

    def test_configured_disk_guard_blocks_checkpoint_and_cleans_lock(self):
        directory=self.root/'budget'; directory.mkdir()
        self.config['checkpoint_size_estimate']=1024**3
        model=DummyModel(); optim=torch.optim.AdamW(model.parameters())
        with patch('training_checkpoint.shutil.disk_usage',return_value=SimpleNamespace(free=9*1024**3-1)):
            with self.assertRaisesRegex(RuntimeError,'insufficient free disk'):
                self.run_model(model,optim,directory,1)
        self.assertEqual(list(directory.iterdir()),[])

    def test_accumulation_resume_across_epoch_boundaries(self):
        self.config['accumulate_microbatches']=2
        def run(model, optim, directory, steps, resume=None, records=None):
            return run_training(model,DummyDiffusion(),optim,self.dataset,run_dir=directory,
                provenance=self.provenance,config=self.config,max_steps=steps,checkpoint_every=2,
                batch_size=2,seed=5,guidance_steps=0,guidance_scale=.5,device=torch.device('cpu'),
                resume_from=resume,accumulate_microbatches=2,
                on_step=lambda metrics,cursor: records.append((metrics,cursor)))
        full=self.root/'accum_full'; full.mkdir()
        split=self.root/'accum_split'; split.mkdir()
        a=DummyModel(); b=DummyModel(); c=DummyModel()
        oa=torch.optim.AdamW(a.parameters(),lr=.01); ob=torch.optim.AdamW(b.parameters(),lr=.01)
        oc=torch.optim.AdamW(c.parameters(),lr=.01)
        first=[]; second=[]
        run(a,oa,full,6,records=first)
        run(b,ob,split,3,records=second)
        run(c,oc,split,6,resume=split/'step_000000003.pt',records=second)
        self.assertEqual(first,second)
        self.assertTrue(torch.equal(a.weight,c.weight))
        self.assertEqual([len(metrics['microbatches']) for metrics,_ in first],[2,1,2,1,2,1])
        for metrics,_ in first:
            sample_ids=[sid for micro in metrics['microbatches'] for sid in micro['sample_ids']]
            self.assertEqual(len(sample_ids),len(set(sample_ids)))
        self.assertEqual([cursor['epoch'] for _,cursor in first],[0,1,1,2,2,3])


if __name__=='__main__': unittest.main()
