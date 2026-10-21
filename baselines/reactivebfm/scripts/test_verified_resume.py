"""Recovery protocol tests use dummy bytes, never robot/model performance data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from prepare_verified_resume import select_verified,recovery_plan,validate_recovery_plan


class RecoveryProtocol(unittest.TestCase):
    def setUp(self):
        self.parent=Path(__file__).resolve().parents[3]/'.aris/tests/verified_resume'
        self.parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.parent);self.run=Path(self.temp.name).resolve()
        (self.run/'checkpoints').mkdir();(self.run/'checkpoints/args.json').write_text('{}')
        self.add_pair(9,10)

    def tearDown(self):
        assert self.run.is_relative_to(self.parent.resolve());self.temp.cleanup()

    def add_pair(self,step,updates):
        rows=[]
        for name in ['model','opt']:
            p=self.run/'checkpoints'/f'{name}{step:09d}.pt';p.write_bytes((name+str(step)).encode())
            rows.append({'file':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        (self.run/f'checkpoint_ready_{step:09d}.json').write_text(json.dumps({'official_step':step,'optimizer_updates':updates,'files':rows,'configuration_sha256':hashlib.sha256(b'{}').hexdigest()}))

    def test_highest_partial_filename_ignored(self):
        (self.run/'checkpoints/model000000999.pt').write_bytes(b'incomplete/unmanifested')
        selected,_=select_verified(self.run);self.assertEqual(selected['optimizer_updates'],10)

    def test_corrupt_latest_pair_rejected(self):
        self.add_pair(19,20);(self.run/'checkpoints/opt000000019.pt').write_bytes(b'changed')
        selected,rejected=select_verified(self.run)
        self.assertEqual(selected['optimizer_updates'],10);self.assertEqual(len(rejected),1)

    def test_no_valid_pair_and_existing_destination_rejected(self):
        with self.assertRaises(FileExistsError):recovery_plan(self.run,self.run,self.run/'checkpoints/args.json')
        (self.run/'checkpoints/model000000009.pt').write_bytes(b'broken')
        with self.assertRaises(RuntimeError):select_verified(self.run)

    def test_plan_does_not_create_output_or_launch(self):
        dest=self.run/'fresh_output';config=self.run/'config.json';config.write_text('{}')
        plan=recovery_plan(self.run,dest,config)
        self.assertFalse(dest.exists());self.assertEqual(plan['status'],'VERIFIED_RECOVERY_PLAN_ONLY')
        self.assertIn('--resume',plan['command'])

    def test_changed_configuration_rejected(self):
        config=self.run/'config.json';config.write_text('{"changed":true}')
        with self.assertRaises(ValueError):recovery_plan(self.run,self.run/'fresh_output',config)

    def test_pair_changed_after_plan_rejected_before_launch(self):
        config=self.run/'config.json';config.write_text('{}')
        plan=recovery_plan(self.run,self.run/'fresh_output',config)
        (self.run/'checkpoints/opt000000009.pt').write_bytes(b'changed after plan')
        with self.assertRaises(RuntimeError):validate_recovery_plan(plan)
        self.assertFalse((self.run/'fresh_output').exists())

    def test_plan_revalidation_is_read_only(self):
        config=self.run/'config.json';config.write_text('{}')
        plan=recovery_plan(self.run,self.run/'fresh_output',config)
        current=validate_recovery_plan(plan)
        self.assertEqual(current['checkpoint']['manifest_sha256'],plan['checkpoint']['manifest_sha256'])
        self.assertFalse((self.run/'fresh_output').exists())


if __name__=='__main__':unittest.main()
