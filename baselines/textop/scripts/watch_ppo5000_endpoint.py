"""Bound one SIGINT after a durable iteration5000 receipt; never load a model.

The producer saves its last complete in-memory learning state to a new durable
emergency file in its exception handler. It may have started the next rollout;
only a separate, post-exit CPU audit can prove the saved state is exactly5000.
"""
import argparse,fcntl,json,os,pathlib,time,traceback
from ppo5000_endpoint_protocol import (RUN,CONTROL_RUN,REQUEST_SHA,ENTRY_SHA,SUP_SHA,RUNTIME_SHA,
 TARGET,boundary,read,sha,write_new,identity,verify_identity,open_bound_pidfd,send_once)
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
sp=pathlib.Path(a.spec);s=read(sp)
assert os.getuid()==1013 and s['run']==CONTROL_RUN and s['authorized'] is True
os.sched_setaffinity(0,{0,1,2,3})
assert sha(__file__)==s['watcher_sha256']
for name,digest in s['bound_sources'].items():assert sha(ROOT/name)==digest
request=ROOT/'receipts'/('step4_train_only_'+RUN+'_request.json')
assert sha(request)==REQUEST_SHA
out=ROOT/'step34_numeric_v1'/CONTROL_RUN;out.mkdir(mode=0o700,exist_ok=False)
lock=(ROOT/'receipts/user_ppo5000_control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
target=s['producer'];supervisor=s['producer_supervisor']
verify_identity(target);verify_identity(supervisor);pidfd=open_bound_pidfd(target)
rp=ROOT/'step34_numeric_v1'/RUN/'receipt.json'
rec=dict(status='WAITING_FOR_COMPLETED_PPO5000_BOUNDARY',run=CONTROL_RUN,
 request_sha256=sha(sp),watcher_sha256=sha(__file__),original_run=RUN,target_iterations=TARGET,
 target_optimizer_updates=TARGET*20,signals_sent=0,checkpoint_payload_reads=0,model_loads=0,
 new_GPU_contexts=0,original_training_source_modified=False,output_writers_closed=False)
def persist():
 tmp=out/'receipt.partial'
 with tmp.open('w') as f:json.dump(rec,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(out/'receipt.json')
start=time.monotonic();persist();last=-1
try:
 while True:
  assert time.monotonic()-start<72*3600,'Observer timed out; never infer production exit or signal'
  verify_identity(target);verify_identity(supervisor)
  producer=read(rp);ready=boundary(producer)
  n=producer['formal_completed_iterations'];rec.update(last_observed_iterations=n,last_observed_updates=producer['formal_optimizer_updates'])
  if ready:
   assert sha(request)==REQUEST_SHA
   for name,digest in s['bound_sources'].items():assert sha(ROOT/name)==digest
   verify_identity(target);verify_identity(supervisor)
   assert boundary(read(rp))
   write_new(out/'producer_before_signal.json',producer)
   # An existing intent forbids another signal even if the outcome is unknown.
   # send_once durably writes the sole intent before invoking pidfd_send_signal.
   send_once(pidfd,target,out/'signal_intent.json',dict(target_iterations=TARGET,
    observed_iterations=n,observed_updates=producer['formal_optimizer_updates'],
    receipt_sha256=sha(rp),checkpoint_not_yet_persisted=True))
   write_new(out/'signal_sent.json',dict(signal='SIGINT',signals_sent=1,target_iterations=TARGET))
   rec.update(status='PPO5000_SIGNAL_SENT_WAITING_FOR_ORIGINAL_EXIT',signals_sent=1);persist();break
  if n//100!=last//100:last=n;persist()
  time.sleep(.1)
 log=ROOT/'step34_numeric_v1'/(RUN+'_supervision');prefix='step4_train_only_'+RUN
 paths=dict(exit=log/'exit.json',process=log/'process_exit.json',outer=ROOT/'receipts'/(prefix+'_outer_exit.json'))
 deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  alive=[]
  for expected in [target,supervisor]:
   try:
    z=identity(expected['pid'])
    if z['start_ticks']==expected['start_ticks'] and z['state'] not in ['Z','X']:alive.append(z)
   except FileNotFoundError:pass
  if not alive and all(p.is_file() for p in paths.values()):break
  time.sleep(1)
 else:raise RuntimeError('One signal already sent; exit unknown, never resend or rearm')
 final=read(rp);ex={k:read(v) for k,v in paths.items()}
 assert final['status']=='STAGE4_ORIGINAL_TRAIN_PPO_FAILED_PARTIAL_RETAINED'
 assert final['error']=="RuntimeError('Owned Stage4 PPO interrupted with signal 2')"
 assert final['request_sha256']==REQUEST_SHA and final['source_sha256']==ENTRY_SHA
 for k in ['exit','process']:
  v=ex[k];assert v['child_reaped'] and v['child_exit_code'] in [1,-2,130] and not v['timed_out']
  assert v['supervisor_error'] is None and v['sample_errors']==[] and not v.get('cleanup_error') and v['shutdown_actions']==[]
 assert ex['outer']['actual_returncode']!=0 and ex['exit']['source_sha256']==SUP_SHA
 assert ex['exit']['request_sha256']==REQUEST_SHA and ex['exit']['receipt_sha256']==sha(rp)
 assert ex['exit']['process_exit_sha256']==sha(paths['process'])
 assert ex['exit']['resource_sha256']==sha(log/'resources.jsonl')
 assert 'emergency_last_good' in final and 'emergency_save_error' not in final
 ck=final['emergency_last_good'];cp=pathlib.Path(ck['path'])
 assert cp==ROOT/'step34_numeric_v1'/RUN/'emergency_last_good.private.pth'
 assert cp.resolve()==cp and cp.stat().st_uid==1013 and not cp.is_symlink()
 assert 0<cp.stat().st_size==ck['bytes']<=128*1024**2 and sha(cp)==ck['sha256']
 common=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(common,fcntl.LOCK_EX|fcntl.LOCK_NB);common.close()
 write_new(out/'producer_final_snapshot.json',final)
 rec.update(status='PPO5000_SIGNAL_AND_SAVED_STATE_AWAITING_CPU_AUDIT',output_writers_closed=True,
  original_exits=ex,original_exit_sha256={k:sha(v) for k,v in paths.items()},
  original_final_receipt_sha256=sha(rp),saved_state=ck,producer_reported_iterations=final['formal_completed_iterations'],
  selected_checkpoint_payload_audited=False,original_nonzero_exits_preserved=True,common_model_lock_released=True)
 persist();print(json.dumps(rec))
except BaseException as e:
 rec.update(status='PPO5000_CONTROL_FAILED_EVIDENCE_RETAINED',error=repr(e),traceback=traceback.format_exc());persist();raise
finally:os.close(pidfd);lock.close()
