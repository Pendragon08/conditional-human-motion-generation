"""CPU-only storage guard witness; never writes a checkpoint or training data."""
import argparse,json,os,pathlib,re,types
from unittest.mock import patch
import checkpoint_storage_guard as guard

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',args.run)
os.sched_setaffinity(0,{0,1,2,3})
out=ROOT/'receipts'/('checkpoint_storage_guard_'+args.run);out.mkdir(exist_ok=False)
policy=json.loads((ROOT/'configs/filter-training-plan.json').read_text())['checkpoint_storage']
checks=[]
def rejects(name,fn):
 try:fn()
 except AssertionError:checks.append(name);return
 raise AssertionError('Expected rejection not observed: '+name)

receipt={'status':'STARTING','cpu_only':True,'new_checkpoints_written':0,'training_started':False,'checks':checks}
try:
 steps=guard.schedule(40000,5000);assert steps==[1000,5000,10000,15000,20000,25000,30000,35000,40000]
 checks.append('fixed_nine_checkpoint_schedule')
 actual=guard.check(out,ROOT,policy,len(steps));checks.append('actual_owned_persistent_writable_destination_and_full_budget')
 with patch.object(guard.shutil,'disk_usage',return_value=types.SimpleNamespace(free=actual['required_bytes'])):
  guard.check(out,ROOT,policy,len(steps));checks.append('exact_full_budget_boundary_passes')
 with patch.object(guard.shutil,'disk_usage',return_value=types.SimpleNamespace(free=actual['required_bytes']-1)):
  rejects('one_byte_below_full_run_budget_rejected',lambda:guard.check(out,ROOT,policy,len(steps)))
 emergency=policy['reserve_bytes']+policy['log_allowance_bytes']+policy['checkpoint_upper_bound_bytes']
 with patch.object(guard.shutil,'disk_usage',return_value=types.SimpleNamespace(free=emergency-1)):
  rejects('unsafe_emergency_save_rejected',lambda:guard.check(out,ROOT,policy,1))
 rejects('outside_personal_task_root_rejected',lambda:guard.check(ROOT.parent,ROOT,policy,1))
 ram=pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
 rejects('volatile_ram_filesystem_rejected',lambda:guard.check(ram,ram,policy,1))
 link=out/'volatile_destination_link';link.symlink_to(ram,target_is_directory=True)
 rejects('symlink_destination_rejected',lambda:guard.check(link,ROOT,policy,1))
 with patch.object(guard.os,'access',return_value=False):
  rejects('unwritable_destination_rejected',lambda:guard.check(out,ROOT,policy,1))
 with patch.object(guard.os,'getuid',return_value=0):
  rejects('unexpected_ownership_context_rejected',lambda:guard.check(out,ROOT,policy,1))
 rejects('missing_destination_rejected',lambda:guard.check(out/'nonexistent',ROOT,policy,1))
 receipt.update(status='CHECKPOINT_STORAGE_GUARD_WITNESS_PASSED',actual_preflight=actual,checks_passed=len(checks),
                source_sha256=__import__('hashlib').sha256(pathlib.Path(guard.__file__).read_bytes()).hexdigest(),
                witness_sha256=__import__('hashlib').sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                output_writers_closed=True)
except BaseException as error:
 receipt.update(status='FAILED',error=repr(error));raise
finally:
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
