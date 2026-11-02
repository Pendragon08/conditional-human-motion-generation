"""One pidfd SIGINT only after real epoch100 atomic save and producer reload.

No model imports, checkpoint deserialization or numerical source mutation.
Original interrupted500-epoch exits and any post100 in-memory work are retained.
"""
import argparse,fcntl,json,os,pathlib,time,traceback
from tmr100_endpoint_protocol import (RUN,CONTROL_RUN,checkpoint_ready,read,sha,write_new,
    verify_checkpoint,identity,verify_identity,open_bound_pidfd,send_once)
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert a.run==CONTROL_RUN and os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3})
sp=ROOT/'configs/tmr100-user-control-spec.json';s=read(sp)
assert s['control_authorized'] and s['schema']=='textop.user_tmr100_endpoint.v1'
assert sha(__file__)==s['entrypoint_sha256']
for name,value in s['bound_sources'].items():assert sha(ROOT/name)==value
original_sp=ROOT/'configs/tmr69d-evaluator-train-spec.json'
assert sha(original_sp)==s['original_spec_sha256']
original=read(original_sp);assert original['epochs']==500 and original['expected_optimizer_updates']==507500
parent=ROOT/'evaluator_tmr_20261029';out=parent/CONTROL_RUN;out.mkdir(mode=0o700,exist_ok=False)
lock=(ROOT/'receipts/user_tmr100_control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
target=s['producer'];verify_identity(target);verify_identity(s['producer_supervisor']);pidfd=open_bound_pidfd(target)
rp=parent/RUN/'receipt.json'
record=dict(status='WAITING_FOR_REAL_TMR_EPOCH100_SAVED_AND_RELOADED',run=CONTROL_RUN,
    source_sha256=sha(__file__),spec_sha256=sha(sp),original_run=RUN,user_target_epochs=100,
    target_optimizer_updates=101500,signals_sent=0,model_loads=0,checkpoint_payload_deserializations=0,
    new_GPU_contexts=0,original_training_source_modified=False,original500_completion_claimed=False,
    output_writers_closed=False)
def save():
    temp=out/'receipt.partial'
    with temp.open('w') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    temp.replace(out/'receipt.json')
started=time.monotonic();save();last=-1
try:
    while True:
        assert time.monotonic()-started<7*24*3600,'Observer timeout never authorizes killing the trainer'
        assert sha(original_sp)==s['original_spec_sha256']
        rec=read(rp);ck=checkpoint_ready(rec,s)
        record.update(last_observed_completed_epochs=rec['completed_epochs'],last_observed_optimizer_updates=rec['formal_optimizer_updates'])
        verify_identity(target);verify_identity(s['producer_supervisor'])
        if ck is not None:
            proof=verify_checkpoint(ck,s['checkpoint_allowed_root'])
            for name,value in s['bound_sources'].items():assert sha(ROOT/name)==value
            again=read(rp);assert checkpoint_ready(again,s)==ck
            write_new(out/'producer_before_signal.json',again)
            context=dict(selected_checkpoint=ck,checkpoint_proof=proof,
                observed_updates_at_request=again['formal_optimizer_updates'],
                original_spec_sha256=s['original_spec_sha256'],original_runtime_binding_sha256=s['original_runtime_binding_sha256'])
            send_once(pidfd,target,out/'signal_intent.json',context)
            write_new(out/'signal_sent.json',dict(signal='SIGINT',signals_sent=1,pidfd_bound=True,**context))
            record.update(status='BOUND_SIGINT_SENT_WAITING_TMR_ORIGINAL_EXITS',signals_sent=1,**context);save();break
        if last!=rec['formal_optimizer_updates']:last=rec['formal_optimizer_updates'];save()
        time.sleep(1)
    log=parent/(RUN+'_supervision');prefix='tmr_evaluator_train_'+RUN
    paths=dict(exit=log/'exit.json',process=log/'process_exit.json',outer=ROOT/'receipts'/(prefix+'_outer_exit.json'))
    deadline=time.monotonic()+300
    while time.monotonic()<deadline:
        alive=[]
        for expected in [s['producer'],s['producer_supervisor']]:
            try:
                actual=identity(expected['pid'])
                if actual['start_ticks']==expected['start_ticks'] and actual['state'] not in ['Z','X']:alive.append(actual)
            except FileNotFoundError:pass
        if not alive and all(p.is_file() for p in paths.values()):break
        time.sleep(1)
    else:raise RuntimeError('One signal already sent; exits unknown. Never resend or rearm.')
    final=read(rp);exits={name:read(path) for name,path in paths.items()}
    assert final['status']=='FAILED_TMR69D_FORMAL_EVALUATOR_ALL_STATES_RETAINED'
    assert final['error']=="RuntimeError('Owned TMR evaluator interrupted with signal 2')",final.get('error')
    assert final['request_sha256']==s['original_spec_sha256'] and final['source_sha256']==s['original_entry_sha256']
    assert any(item==ck for item in final['checkpoints'])
    assert exits['outer']['actual_returncode']!=0
    for name in ['exit','process']:
        ex=exits[name];assert ex['child_reaped'] and ex['child_exit_code'] in [1,-2,130] and not ex['timed_out']
        assert ex['supervisor_error'] is None and ex['sample_errors']==[] and not ex.get('cleanup_error') and ex['shutdown_actions']==[]
    assert exits['exit']['receipt_sha256']==sha(rp)
    assert exits['exit']['source_sha256']==s['original_supervisor_sha256']
    assert exits['exit']['request_sha256']==s['original_spec_sha256']
    assert exits['exit']['process_exit_sha256']==sha(paths['process'])
    resources=log/'resources.jsonl';rows=[json.loads(x) for x in resources.read_text().splitlines()]
    assert sha(resources)==exits['exit']['resource_sha256']
    assert len(rows)==exits['exit']['samples_written']==exits['process']['samples_written']
    assert rows[-1]['phase']=='after_exit' and rows[-1]['owned_group_processes']==[]
    verify_checkpoint(ck,s['checkpoint_allowed_root'])
    common=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(common,fcntl.LOCK_EX|fcntl.LOCK_NB);common.close()
    write_new(out/'producer_final_snapshot.json',final)
    record.update(status='USER_TMR100_EPOCH_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED',output_writers_closed=True,
        selected_epochs=100,selected_optimizer_updates=101500,selected_checkpoint=ck,original_exit_records=exits,
        original_exit_sha256={name:sha(path) for name,path in paths.items()},original_resources_sha256=sha(resources),
        original_final_receipt_sha256=sha(rp),post_target_in_memory_updates_not_selected=max(0,final['formal_optimizer_updates']-101500),
        original_post_target_emergency_not_selected=True,common_model_lock_released=True)
    save();print(json.dumps(record))
except BaseException as error:
    record.update(status='FAILED_TMR100_ENDPOINT_RETAIN_ALL_EVIDENCE',error=repr(error),traceback=traceback.format_exc());save();raise
finally:os.close(pidfd);lock.close()
