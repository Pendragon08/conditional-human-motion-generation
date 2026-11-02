"""Wait for the actual100k saved/reloaded state, then one identity-bound SIGINT.

No training file is edited. The original200k program may report KeyboardInterrupt
and a nonzero exit; preserve that fact and all post-target memory observations.
Only the exact100k immutable checkpoint is selected for the new handoff path.
"""
import argparse,fcntl,hashlib,json,os,pathlib,time,traceback
from vae100k_control_protocol import read,sha,write_new,checkpoint_ready,verify_checkpoint,open_bound_pidfd,send_once,identity,verify_identity
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert a.run=='user_vae100k_control_v1' and os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3})
sp=ROOT/'configs/vae100k-user-control-spec.json';s=read(sp)
assert s['schema']=='textop.user_vae100k_control.v1' and s['control_authorized'] and s['target_primitive_attempts']==100000
assert sha(__file__)==s['entrypoint_sha256']
for n,h in s['bound_sources'].items():assert sha(ROOT/n)==h,n
assert sha(ROOT/'configs/filtered15000-vae-spec.json')==s['original_spec_sha256']=='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
out=ROOT/'vae100k_user_handoff_20261028'/a.run;out.mkdir(parents=True,exist_ok=False)
control=(ROOT/'receipts/vae100k_user_control.lock').open('a+');fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
target=s['producer'];verify_identity(target);pidfd=open_bound_pidfd(target)
rp=ROOT/'filtered15000_20261027/vae_runs'/s['original_run']/'receipt.json'
record=dict(status='WAITING_FOR_REAL_VAE100000_SAVED_AND_RELOADED',run=a.run,user_requested_target=100000,
 original_run=s['original_run'],source_sha256=sha(__file__),spec_sha256=sha(sp),signals_sent=0,
 new_model_loads=0,new_gpu_contexts=0,new_sdk_tasks=0,original_training_source_modified=False)
def save():
 temp=out/'receipt.partial';temp.write_text(json.dumps(record,indent=2,allow_nan=False));temp.replace(out/'receipt.json')
save();started=time.monotonic();last=-1
try:
 while True:
  assert time.monotonic()-started<6*3600,'Control timeout; trainer is not killed on observer timeout'
  assert sha(ROOT/'configs/filtered15000-vae-spec.json')==s['original_spec_sha256']
  r=read(rp);record['last_observed_primitive_attempts']=r['completed_primitive_attempts']
  assert r['status']=='RUNNING_FILTERED15000_VAE_TRAINING','Producer is no longer running; inspect real exits, do not signal'
  ck=checkpoint_ready(r,s)
  if ck is not None:
   proof=verify_checkpoint(ck,s['checkpoint_allowed_root']);verify_identity(target)
   again=read(rp);assert again['status']=='RUNNING_FILTERED15000_VAE_TRAINING' and checkpoint_ready(again,s)==ck
   write_new(out/'producer_before_signal.json',again)
   context=dict(checkpoint=ck,checkpoint_proof=proof,producer_receipt_sha256=sha(rp),
    original_spec_sha256=s['original_spec_sha256'],original_runtime_binding_sha256=s['original_runtime_binding_sha256'],
    observed_primitive_attempts_at_request=again['completed_primitive_attempts'],original200k_completion_claimed=False)
   send_once(pidfd,target,out/'signal_intent.json',context)
   write_new(out/'signal_sent.json',dict(signal='SIGINT',pidfd_bound=True,signals_sent=1,sent_unix=time.time(),**context))
   record.update(status='BOUND_SIGINT_SENT_WAITING_ORIGINAL_EXITS',signals_sent=1,selected_checkpoint=ck,checkpoint_proof=proof,
    observed_primitive_attempts_at_request=again['completed_primitive_attempts']);save();break
  verify_identity(target)
  if r['completed_primitive_attempts']!=last:
   last=r['completed_primitive_attempts'];save()
  time.sleep(1)
 deadline=time.monotonic()+240
 prefix='filtered15000_vae_'+s['original_run']
 paths={k:ROOT/'receipts'/(prefix+x) for k,x in [('exit','_exit.json'),('process','_process_exit.json'),('outer','_outer_exit.json')]}
 while time.monotonic()<deadline:
  alive=[]
  for expected in [s['producer'],s['supervisor']]:
   try:
    actual=identity(expected['pid'])
    if actual['start_ticks']==expected['start_ticks'] and actual['state'] not in ['Z','X']:alive.append(actual)
   except FileNotFoundError:pass
  if not alive and all(v.is_file() for v in paths.values()):break
  time.sleep(1)
 else:raise RuntimeError('Signal already sent once; original exit still unknown, preserve intent and do not resend')
 exits={k:read(v) for k,v in paths.items()};final=read(rp)
 assert final['status']=='FAILED_FILTERED15000_VAE' and final.get('error')=='KeyboardInterrupt()',final.get('error')
 for k in ['exit','process']:
  v=exits[k];assert v['child_reaped'] and not v['timed_out'] and v['supervisor_error'] is None
  assert v['sample_errors']==[] and not v.get('cleanup_error')
  assert v['child_exit_code'] in [-2,1,130]
 assert exits['outer']['actual_returncode']!=0,'Do not rewrite original non-200k termination as normal200k completion'
 assert any(x==ck for x in final['checkpoints'])
 # SIGINT can land in a post-target primitive. Its memory clock is discarded;
 # never repair it or demand that it describe the already-closed100k state.
 for k,sk in [('spec_sha256','original_spec_sha256'),('source_sha256','original_entry_sha256'),('runtime_binding_sha256','original_runtime_binding_sha256')]:assert final[k]==s[sk]
 verify_checkpoint(ck,s['checkpoint_allowed_root'])
 verify_lock=(ROOT/'receipts/full_library_capacity.lock').open('a+')
 fcntl.flock(verify_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_lock.close()
 record.update(status='USER_VAE100000_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED',output_writers_closed=True,
  original_exit_records=exits,original_exit_sha256={k:sha(v) for k,v in paths.items()},
  original_final_receipt_sha256=sha(rp),original_final_attempts=final['completed_primitive_attempts'],
  post_target_in_memory_attempts_not_selected=max(0,final['completed_primitive_attempts']-100000),
  original_memory_clock_after_interrupt_not_accepted_as_checkpoint=True,
  common_lock_released=True,selected_checkpoint_attempts=100000,original200k_completion_claimed=False)
 write_new(out/'producer_final_snapshot.json',final);save();print(json.dumps(record),flush=True)
except BaseException as error:
 record.update(status='FAILED_VAE100K_CONTROL_RETAIN_ALL_EVIDENCE',error=repr(error),traceback=traceback.format_exc());save();raise
finally:os.close(pidfd);control.close()
