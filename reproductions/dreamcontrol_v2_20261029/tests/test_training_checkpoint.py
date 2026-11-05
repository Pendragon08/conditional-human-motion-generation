import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from training_checkpoint import save_checkpoint, load_checkpoint, _BoundedWriter


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name)/'checkpoint.pt'
        self.provenance = {'dataset_manifest_sha256':'a'*64,'stats_sha256':'b'*64,'model_init_sha256':'c'*64}
        self.config = {'batch_size':2,'lr':1e-5}
        self.cursor = {'step':7,'epoch':1,'next_batch':3}
        self.model = torch.nn.Linear(2,1)
        self.opt = torch.optim.AdamW(self.model.parameters(),lr=.01)
        self.opt.zero_grad(); self.model(torch.ones(1,2)).sum().backward(); self.opt.step()

    def save(self):
        save_checkpoint(self.file,self.model,self.opt,provenance=self.provenance,config=self.config,cursor=self.cursor)

    def test_roundtrip_restores_model_optimizer_and_rng(self):
        random.seed(11); np.random.seed(12); torch.manual_seed(13)
        self.save()
        expected = (random.random(),float(np.random.random()),torch.rand(1).item())
        model = torch.nn.Linear(2,1)
        opt = torch.optim.AdamW(model.parameters(),lr=.1)
        cursor = load_checkpoint(self.file,model,opt,expected_provenance=self.provenance,expected_config=self.config)
        actual = (random.random(),float(np.random.random()),torch.rand(1).item())
        self.assertEqual(cursor,self.cursor)
        self.assertEqual(actual,expected)
        self.assertEqual(opt.param_groups[0]['lr'],.01)
        for a,b in zip(self.model.parameters(),model.parameters()): self.assertTrue(torch.equal(a,b))
        self.assertEqual(len(opt.state),len(self.opt.state))
        for m, o in ((self.model, self.opt), (model, opt)):
            o.zero_grad(); m(torch.tensor([[.3, -.4]])).square().sum().backward(); o.step()
        for a,b in zip(self.model.parameters(),model.parameters()): self.assertTrue(torch.equal(a,b))

    def test_no_overwrite_and_reject_mismatch(self):
        self.save()
        original = self.file.read_bytes()
        with self.assertRaises(FileExistsError): self.save()
        self.assertEqual(self.file.read_bytes(),original)
        before = self.model.weight.detach().clone()
        with self.assertRaisesRegex(ValueError,'mismatch'):
            load_checkpoint(self.file,self.model,self.opt,expected_provenance={**self.provenance,'stats_sha256':'d'*64},expected_config=self.config)
        self.assertTrue(torch.equal(before,self.model.weight))
        self.assertEqual(list(Path(self.tmp.name).glob('.checkpoint-*.tmp')),[])

    def test_reject_bad_provenance_before_publishing(self):
        with self.assertRaises(ValueError):
            save_checkpoint(self.file,self.model,self.opt,provenance={},config=self.config,cursor=self.cursor)
        self.assertFalse(self.file.exists())

    def test_disk_guard_rejects_before_write_and_preserves_files(self):
        with patch('training_checkpoint.shutil.disk_usage',return_value=SimpleNamespace(free=109)):
            with self.assertRaisesRegex(RuntimeError,'insufficient free disk'):
                save_checkpoint(self.file,self.model,self.opt,provenance=self.provenance,config=self.config,
                                cursor=self.cursor,max_checkpoint_bytes=10,disk_floor_bytes=100)
        self.assertFalse(self.file.exists())
        self.assertEqual(list(self.file.parent.glob('.checkpoint-*.tmp')),[])

    def test_disk_guard_accepts_sufficient_space_and_caps_checkpoint(self):
        with patch('training_checkpoint.shutil.disk_usage',return_value=SimpleNamespace(free=10**9)):
            save_checkpoint(self.file,self.model,self.opt,provenance=self.provenance,config=self.config,
                            cursor=self.cursor,max_checkpoint_bytes=10**7,disk_floor_bytes=100)
        self.assertTrue(self.file.exists())
        original=self.file.read_bytes()
        other=self.file.parent/'small.pt'
        with patch('training_checkpoint.shutil.disk_usage',return_value=SimpleNamespace(free=10**9)):
            with self.assertRaises(Exception):
                save_checkpoint(other,self.model,self.opt,provenance=self.provenance,config=self.config,
                                cursor=self.cursor,max_checkpoint_bytes=10,disk_floor_bytes=100)
        self.assertEqual(self.file.read_bytes(),original)
        self.assertFalse(other.exists())
        self.assertEqual(list(self.file.parent.glob('.checkpoint-*.tmp')),[])

    def test_disk_guard_stops_if_free_space_drops_during_write(self):
        spaces=[SimpleNamespace(free=10**9), SimpleNamespace(free=100)]
        with patch('training_checkpoint.shutil.disk_usage',side_effect=spaces):
            with self.assertRaises(Exception):
                save_checkpoint(self.file,self.model,self.opt,provenance=self.provenance,config=self.config,
                                cursor=self.cursor,max_checkpoint_bytes=10**7,disk_floor_bytes=100)
        self.assertFalse(self.file.exists())
        self.assertEqual(list(self.file.parent.glob('.checkpoint-*.tmp')),[])

    def test_bounded_writer_completes_partial_writes(self):
        class Partial:
            def __init__(self): self.data=bytearray()
            def tell(self): return len(self.data)
            def write(self, data):
                chunk=data[:2]; self.data.extend(chunk); return len(chunk)
        stream=Partial()
        with patch('training_checkpoint.shutil.disk_usage',return_value=SimpleNamespace(free=1000)):
            self.assertEqual(_BoundedWriter(stream,self.file.parent,20,100).write(b'abcdef'),6)
        self.assertEqual(stream.data,b'abcdef')


if __name__ == '__main__':
    unittest.main()
