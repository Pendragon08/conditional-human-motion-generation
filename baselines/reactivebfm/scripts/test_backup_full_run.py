"""Safety unit tests with dummy bytes, not scientific/robot experiment results."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import shutil
import subprocess
import sys
from backup_full_run import Worker, digest, freeze_package


class CleanupProtocol(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[3] / '.aris/tests/backup_worker'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(parent.resolve())
        self.allowed_parent = parent.resolve()
        cps = self.root / 'checkpoints'; cps.mkdir()
        self.old = cps / 'model000000009.pt'; self.old.write_bytes(b'dummy owned unit-test checkpoint')
        self.latest = cps / 'model000000019.pt'; self.latest.write_bytes(b'newer dummy unit-test checkpoint')
        row = {'file': str(self.old), 'bytes': self.old.stat().st_size, 'sha256': hashlib.sha256(self.old.read_bytes()).hexdigest()}
        self.older = {'official_step': 9, 'optimizer_updates': 10, 'files': [row]}
        self.newest = {'official_step': 19, 'optimizer_updates': 20, 'files': []}
        self.assets = {self.old.name: {'size': row['bytes'], 'digest': 'sha256:' + row['sha256']}}
        self.worker = Worker.__new__(Worker)
        self.worker.c = {'server_output': str(self.root)}
        self.worker.state = {'processed': {'9': {}}}
        self.called = False
        def remote(script, timeout=120):
            self.called = True
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream): exec(script, {})
            return stream.getvalue()
        self.worker.remote = remote
        self.worker.log = lambda *args, **kwargs: None

    def tearDown(self):
        # Verify the absolute recursive cleanup target remains in test workspace.
        assert self.root.resolve().is_relative_to(self.allowed_parent)
        self.temp.cleanup()

    def test_wrong_github_digest_never_reaches_deletion(self):
        self.assets[self.old.name]['digest'] = 'sha256:' + '0' * 64
        with self.assertRaises(RuntimeError):
            self.worker.cleanup_server_older(self.newest, [self.older], self.assets)
        self.assertFalse(self.called); self.assertTrue(self.old.exists())

    def test_changed_server_bytes_are_preserved(self):
        self.old.write_bytes(b'tampered after backup')
        with self.assertRaises(RuntimeError):
            self.worker.cleanup_server_older(self.newest, [self.older], self.assets)
        self.assertTrue(self.old.exists()); self.assertTrue(self.latest.exists())

    def test_valid_old_copy_removed_latest_kept(self):
        self.worker.cleanup_server_older(self.newest, [self.older, self.newest], self.assets)
        self.assertFalse(self.old.exists()); self.assertTrue(self.latest.exists())

    def test_path_outside_checkpoint_directory_preserved(self):
        outside = self.root / self.old.name; outside.write_bytes(self.old.read_bytes())
        self.older['files'][0]['file'] = str(outside)
        with self.assertRaises(RuntimeError):
            self.worker.cleanup_server_older(self.newest, [self.older], self.assets)
        self.assertTrue(outside.exists()); self.assertTrue(self.latest.exists())

    def test_package_bytes_stay_immutable_after_retry(self):
        directory = self.root / 'evidence'; directory.mkdir()
        (directory / 'status.json').write_text('{"updates":10}')
        package = directory / 'evidence.tar.gz'
        freeze_package(directory, package); first = digest(package)
        (directory / 'status.json').write_text('{"updates":999}')
        (directory / 'new.txt').write_text('later snapshot')
        freeze_package(directory, package)
        self.assertEqual(first, digest(package))

    def test_missing_remote_receipt_is_failed_and_never_rerun(self):
        evaluation_root = self.root / 'evaluations'; evaluation_root.mkdir()
        evidence = self.root / 'evaluation_evidence'; evidence.mkdir()
        self.worker.c.update(python='unused', runtime_root='unused', evaluation_root=str(evaluation_root))
        self.worker.ssh = ['unused-ssh']
        self.worker.download = lambda source, destination: shutil.copyfile(source, destination)
        manifest = {'optimizer_updates':1000, 'files':[{'file':'model000000999.pt'}]}
        with patch('backup_full_run.subprocess.run', return_value=subprocess.CompletedProcess([],1,'','intentional test failure')) as mocked:
            first = self.worker.evaluate(manifest, evidence)
            archives = {p.name:digest(p) for p in evidence.glob('*.tar.gz')}
            second = self.worker.evaluate(manifest, evidence)
        self.assertEqual(mocked.call_count, 9)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertTrue(all(x['status']=='EVALUATION_LAUNCH_FAILED' and x['physics_evaluated'] is False for x in first))
        self.assertEqual(archives, {p.name:digest(p) for p in evidence.glob('*.tar.gz')})

    def test_existing_download_is_verified_against_remote_bytes(self):
        source=self.root/'remote_receipt.json';source.write_text('{"status":"FAILED"}')
        local=self.root/'local_receipt.json';local.write_text('{"status":"FAKE_PASS"}')
        with self.assertRaises(RuntimeError):
            self.worker.download(str(source),local)
        self.assertEqual(local.read_text(),'{"status":"FAKE_PASS"}')

    def test_changed_cached_evaluation_receipt_is_rejected(self):
        from backup_full_run import seal_receipt, check_receipt
        path=self.root/'evaluation_receipt.json';path.write_text('{"status":"FAILED"}')
        seal_receipt(path);check_receipt(path)
        path.write_text('{"status":"PHYSICAL_CLOSED_LOOP_EXECUTED"}')
        with self.assertRaises(RuntimeError):check_receipt(path)

    def test_release_asset_enumeration_is_paginated(self):
        self.worker.c.update(github_repo='owner/repo',release_tag='test')
        calls=[]
        def call(args):
            endpoint=args[-1];calls.append(endpoint)
            if '/tags/' in endpoint:return '{"id":123}'
            if endpoint.endswith('&page=1'):return json.dumps([{'name':f'asset_{i}'} for i in range(100)])
            if endpoint.endswith('&page=2'):return '[{"name":"asset_100"}]'
            raise AssertionError(endpoint)
        self.worker.call=call
        assets=self.worker.assets();self.assertEqual(len(assets),101);self.assertEqual(len(calls),3)

    def test_optimized_interpreter_is_rejected(self):
        for module in ['backup_full_run','prepare_verified_resume']:
            code='import sys;sys.path.insert(0,'+repr(str(Path(__file__).resolve().parent))+');import '+module
            r=subprocess.run([sys.executable,'-O','-c',code],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('Optimized Python',r.stderr)

    def test_completion_requires_both_exact_ranks(self):
        self.worker.root=self.root;self.worker.state={'processed':{'1000000':{}}}
        self.worker.assets=lambda:{}
        self.worker.upload=lambda *args:None
        manifest={'official_step':1000000,'optimizer_updates':1000000}
        ranks=[{'rank':0,'world_size':2,'status':'FULL_SCHEDULE_COMPLETED','full_training_completed':True,'optimizer_updates':1000000}]
        self.worker.snapshot=lambda:{'statuses':ranks,'manifests':[manifest],'utc_epoch':2000,'training_process_running':False}
        events=[];self.worker.log=lambda name,**kwargs:events.append(name)
        self.worker.run(once=True)
        self.assertIn('incomplete_status_rank_set_no_completion',events)
        self.assertNotIn('full_schedule_backed_up_and_diagnostics_recorded',events)
        ranks.append({**ranks[0],'rank':1});events.clear();self.worker.run(once=True)
        self.assertIn('full_schedule_backed_up_and_diagnostics_recorded',events)


if __name__ == '__main__': unittest.main()
