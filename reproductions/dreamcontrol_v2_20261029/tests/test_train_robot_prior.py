import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import train_robot_prior as entry


class PriorEntryTests(unittest.TestCase):
    def test_cannot_run_when_feature_audit_is_not_physical_acceptance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'run').mkdir(); (root/'manifest.json').write_text('{"records": []}')
            (root/'initial.pt').write_bytes(b'fixture')
            np.savez(root/'stats.npz',feature_mean=np.zeros(263),feature_std=np.ones(263),
                     spatial_mean=np.zeros(66),spatial_std=np.ones(66))
            cfg={'dataset_root':str(root),'manifest':str(root/'manifest.json'),'stats':str(root/'stats.npz'),
                 'initial_checkpoint':str(root/'initial.pt'),'run_dir':str(root/'run'),'min_std':.001,
                 'batch_size':2,'seed':1,'max_steps':10,'checkpoint_every':5,'guidance_steps':1,
                 'guidance_scale':.5,'learning_rate':1e-5,'weight_decay':0.,'device':'cuda:0',
                 'checkpoint_size_estimate':1073741824}
            (root/'config.json').write_text(json.dumps(cfg))
            report={'technical_checks_passed':True,'training_ready':False}
            with patch.object(sys,'argv',['train_robot_prior','--config',str(root/'config.json'),'--run']), \
                 patch.object(entry,'audit',return_value=report), \
                 patch('audit_ssm_source_evidence.audit_ssm_source_bundle',return_value={'training_ready':False}), \
                 patch.object(entry,'resource_preflight') as resource, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'training blocked'): entry.main()
                resource.assert_not_called()
            self.assertEqual(list((root/'run').iterdir()),[])

    def test_occupied_gpu_and_insufficient_disk_rejected(self):
        fake_disk=SimpleNamespace(free=10**12)
        responses=['0, GPU-foo, 40000\n','GPU-foo, 1234\n']
        with patch.object(entry.shutil,'disk_usage',return_value=fake_disk), \
             patch.object(entry.subprocess,'check_output',side_effect=responses):
            with self.assertRaisesRegex(RuntimeError,'existing compute task'):
                entry.resource_preflight(torch.device('cuda:0'),Path('.'),checkpoint_size_estimate=1073741824,
                                         max_steps=10,checkpoint_every=5)

    def test_checkpoint_budget_preserves_eight_gib_floor(self):
        gib = 1024**3
        # Two planned checkpoints and two safety allowances require 4 GiB,
        # in addition to the 8 GiB storage floor.
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=12*gib-1)), \
             patch.object(entry.subprocess,'check_output') as query:
            with self.assertRaisesRegex(RuntimeError,'insufficient free disk'):
                entry.resource_preflight(torch.device('cuda:0'),Path('.'),checkpoint_size_estimate=gib,
                                         max_steps=10,checkpoint_every=5)
            query.assert_not_called()
        responses=['0, GPU-foo, 40000\n','']
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=12*gib)), \
             patch.object(entry.subprocess,'check_output',side_effect=responses):
            entry.resource_preflight(torch.device('cuda:0'),Path('.'),checkpoint_size_estimate=gib,
                                     max_steps=10,checkpoint_every=5)

    def test_shared_gpu_requires_explicit_small_fraction_and_headroom(self):
        fake_disk=SimpleNamespace(free=10**12)
        responses=['1, GPU-bar, 40000\n','GPU-bar, 9876\n','1, 48000\n']
        with patch.dict(os.environ,{},clear=True), patch.object(entry.shutil,'disk_usage',return_value=fake_disk), \
             patch.object(entry.subprocess,'check_output',side_effect=responses):
            entry.resource_preflight(torch.device('cuda:1'),Path('.'),checkpoint_size_estimate=1073741824,
                                     max_steps=10,checkpoint_every=5,allow_shared_gpu=True,gpu_memory_fraction=.1)
        with patch.dict(os.environ,{},clear=True), patch.object(entry.shutil,'disk_usage',return_value=fake_disk):
            for fraction in (None,0,.11,float('nan'),True):
                with self.assertRaisesRegex(ValueError,'memory fraction'):
                    entry.resource_preflight(torch.device('cuda:1'),Path('.'),checkpoint_size_estimate=1073741824,
                                             max_steps=10,checkpoint_every=5,allow_shared_gpu=True,gpu_memory_fraction=fraction)
        responses=['1, GPU-bar, 11000\n','GPU-bar, 9876\n','1, 100000\n']
        with patch.dict(os.environ,{},clear=True), patch.object(entry.shutil,'disk_usage',return_value=fake_disk), \
             patch.object(entry.subprocess,'check_output',side_effect=responses):
            with self.assertRaisesRegex(RuntimeError,'headroom'):
                entry.resource_preflight(torch.device('cuda:1'),Path('.'),checkpoint_size_estimate=1073741824,
                                         max_steps=10,checkpoint_every=5,allow_shared_gpu=True,gpu_memory_fraction=.1)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'},clear=True), patch.object(entry.shutil,'disk_usage',return_value=fake_disk):
            with self.assertRaisesRegex(ValueError,'CUDA_VISIBLE_DEVICES unset'):
                entry.resource_preflight(torch.device('cuda:0'),Path('.'),checkpoint_size_estimate=1073741824,
                                         max_steps=10,checkpoint_every=5,allow_shared_gpu=True,gpu_memory_fraction=.1)
        with patch.object(entry.shutil,'disk_usage',return_value=SimpleNamespace(free=100)):
            with self.assertRaisesRegex(RuntimeError,'insufficient free disk'):
                entry.resource_preflight(torch.device('cuda:0'),Path('.'),checkpoint_size_estimate=1073741824,
                                         max_steps=10,checkpoint_every=5)


if __name__=='__main__': unittest.main()
