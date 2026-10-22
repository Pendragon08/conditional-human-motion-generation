"""Safety unit tests with dummy bytes, not scientific/robot experiment results."""
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import tarfile
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

    def test_recovery_trace_is_frozen_at_checkpoint_cutoff(self):
        self.worker.c['capture_recovery_trace'] = True
        rows=[{'loop_step':9,'optimizer_updates':10},{'loop_step':19,'optimizer_updates':20},
              {'loop_step':20,'optimizer_updates':21}]
        for row in rows:row.update(step_validated=True,ema_validated=True,optimizer_step_calls=row['optimizer_updates'],validated_optimizer_updates=row['optimizer_updates'])
        log=self.root/'health_rank0.jsonl'
        log.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        (self.root/'health_rank1.jsonl').write_bytes(log.read_bytes())
        for name in ['recovery_launch.json','source_pins_verified.json','status_rank0.json','status_rank1.json']:
            (self.root/name).write_text('{}')
        destination=self.root/'trace_copy';destination.mkdir()
        self.worker.download=lambda source,target:shutil.copyfile(source,target)
        self.worker.capture_recovery_trace(self.newest,destination)
        archive=destination/'training_trace_000000019.tar.gz';before=digest(archive)
        with tarfile.open(archive) as tar:
            captured=[json.loads(x) for x in tar.extractfile('health_rank0_until_checkpoint.jsonl').read().splitlines()]
        self.assertEqual([x['optimizer_updates'] for x in captured],[10,20])
        with log.open('a') as stream:stream.write(json.dumps({'loop_step':21,'optimizer_updates':22})+'\n')
        self.worker.capture_recovery_trace(self.newest,destination)
        self.assertEqual(before,digest(archive))

    def test_trace_refuses_missing_rank_and_unvalidated_endpoint(self):
        self.worker.c['capture_recovery_trace']=True
        destination=self.root/'trace_copy';destination.mkdir()
        self.worker.download=lambda source,target:shutil.copyfile(source,target)
        for name in ['recovery_launch.json','source_pins_verified.json','status_rank0.json','status_rank1.json']:
            (self.root/name).write_text('{}')
        row={'loop_step':19,'optimizer_updates':20,'step_validated':True,'ema_validated':True,'optimizer_step_calls':20,'validated_optimizer_updates':20}
        (self.root/'health_rank0.jsonl').write_text(json.dumps(row)+'\n')
        with self.assertRaises(RuntimeError):self.worker.capture_recovery_trace(self.newest,destination)
        row['ema_validated']=False
        (self.root/'health_rank1.jsonl').write_text(json.dumps(row)+'\n')
        with self.assertRaises(RuntimeError):self.worker.capture_recovery_trace(self.newest,destination)
        self.assertFalse((self.root/'training_trace_000000019.tar.gz').exists())

    def test_evaluation_model_identity_rejects_changed_bytes(self):
        row=self.older['files'][0]
        self.worker.verify_model_identity(row)
        self.old.write_bytes(b'changed after manifest')
        with self.assertRaises(RuntimeError):self.worker.verify_model_identity(row)

    def test_postcheck_failure_receipt_cannot_be_promoted_after_restore(self):
        evaluation_root=self.root/'evaluations';evaluation_root.mkdir()
        evidence=self.root/'evaluation_evidence';evidence.mkdir()
        self.worker.c.update(python='unused',runtime_root='unused',evaluation_root=str(evaluation_root),bind_evaluation_model_sha256=True)
        self.worker.ssh=['unused-ssh']
        self.worker.download=lambda source,destination:shutil.copyfile(source,destination)
        row=self.older['files'][0];original=self.old.read_bytes()
        manifest={'optimizer_updates':1000,'files':[row]}
        def failed_trial(*args,**kwargs):
            self.old.write_bytes(b'wrong model bytes during trial')
            return subprocess.CompletedProcess([],1,'','intentional test failure')
        with patch('backup_full_run.subprocess.run',side_effect=failed_trial):
            with self.assertRaises(RuntimeError):self.worker.evaluate(manifest,evidence)
        orphan=next(evidence.glob('*_receipt.json'));before=digest(orphan)
        self.assertFalse(list(evidence.glob('*_checkpoint_identity.json')))
        self.old.write_bytes(original)
        with patch('backup_full_run.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','intentional test failure')) as mocked:
            results=self.worker.evaluate(manifest,evidence)
        self.assertEqual(mocked.call_count,9)
        self.assertEqual(len(results),9)
        self.assertEqual(before,digest(orphan))
        self.assertFalse(orphan.with_name(orphan.name.replace('_receipt.json','_checkpoint_identity.json')).exists())
        bindings=list(evidence.glob('*_checkpoint_identity.json'));self.assertEqual(len(bindings),9)
        self.assertTrue(any('_retry001_' in p.name for p in bindings))
        for p in bindings:
            binding=json.loads(p.read_text());self.assertEqual(binding['model']['sha256'],row['sha256'])

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
