"""CPU-only isolated boundary and pidfd test; signals only its own child."""
import argparse,copy,json,os,pathlib,subprocess,sys
from ppo5000_endpoint_protocol import boundary,read,write_new,identity,verify_identity,open_bound_pidfd,send_once,RUN,TARGET
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
out=pathlib.Path(a.out);assert out.parent.resolve()==out.parent and out.parent.stat().st_uid==os.getuid()==1013
out.mkdir(mode=0o700,exist_ok=False)
original=read(ROOT/'step34_numeric_v1'/RUN/'receipt.json')
synthetic=copy.deepcopy(original);synthetic.update(formal_completed_iterations=TARGET,formal_optimizer_updates=20*TARGET,
 optimizer_attempts_in_process=20*TARGET,actual_optimizer_calls_in_process=20*TARGET,checkpoints=[])
assert boundary(synthetic) is True
waiting=copy.deepcopy(synthetic);waiting.update(formal_completed_iterations=TARGET-1,formal_optimizer_updates=(TARGET-1)*20,
 optimizer_attempts_in_process=(TARGET-1)*20,actual_optimizer_calls_in_process=(TARGET-1)*20)
assert boundary(waiting) is False
changes=[dict(formal_completed_iterations=TARGET+1),dict(formal_completed_iterations=True),
 dict(formal_optimizer_updates=TARGET),dict(optimizer_attempts_in_process=100001),
 dict(actual_optimizer_calls_in_process=99999),dict(request_sha256='0'*64),dict(source_sha256='0'*64),
 dict(status='FAILED'),dict(generated_training_clips=1),dict(original_VAL_motions_in_training=1),
 dict(old_teacher_training_updates=1),dict(output_writers_closed=True),dict(checkpoints=[{}]),dict(emergency_last_good={})]
for change in changes:
 bad=copy.deepcopy(synthetic);bad.update(change)
 try:boundary(bad)
 except (AssertionError,KeyError,TypeError):pass
 else:raise AssertionError('Invalid synthetic metadata accepted: '+repr(change))
with (out/'child.log').open('xb') as stream:
 child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 fd=None
 try:
  expected=identity(child.pid);assert any('time.sleep(60)' in x for x in expected['argv'])
  for bad in [dict(expected,start_ticks=str(int(expected['start_ticks'])+1)),dict(expected,argv=expected['argv']+['wrong'])]:
   try:verify_identity(bad)
   except AssertionError:pass
   else:raise AssertionError('Wrong identity accepted')
  fd=open_bound_pidfd(expected);existing=out/'existing_intent.json';write_new(existing,dict(status='UNKNOWN_NO_RESEND'))
  try:send_once(fd,expected,existing,dict(fixture_only=True))
  except FileExistsError:pass
  else:raise AssertionError('Duplicate signal intent accepted')
  assert child.poll() is None
  send_once(fd,expected,out/'signal_intent.json',dict(fixture_only=True))
  code=child.wait(timeout=10);assert code!=0
 finally:
  if fd is not None:os.close(fd)
  if child.poll() is None:child.terminate();child.wait(timeout=10)
  stream.flush();os.fsync(stream.fileno())
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
result=dict(status='PPO5000_BOUNDARY_CPU_FIXTURE_VERIFIED',metadata_negative_cases=len(changes),
 identity_negative_cases=2,unknown_duplicate_intent_rejected=True,fixture_signals=1,
 production_signals=0,fixture_child_exit=code,fixture_child_reaped=True,model_loads=0,GPU_contexts=0,
 real_checkpoint_reads=0,actual_training_updates=0)
write_new(out/'receipt.json',result);print(json.dumps(result))
