"""User-authorized15k direct full-library physics boundary; never edit trainer or signal a PID number."""
import argparse,json,os,pathlib,signal,time
from filter_recovery_backup_protocol import RUN,preflight,sha
from quick_pidfd import BoundProcess,start_ticks
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')

def validate_identity(s,launch,live,members):
    out=str(ROOT/'tracker_validation'/RUN);trainer=str(ROOT/'scripts/train_filter_policy.py')
    cmd=launch['command'];assert launch['run']==RUN and launch['spec_sha256']==s['resume_spec_sha256']
    assert trainer in cmd and cmd[cmd.index('--output')+1]==out and cmd[cmd.index('--mode')+1]=='formal'
    assert cmd[cmd.index('--device')+1]=='cuda:1' and '--resume' in cmd
    group=live['child_pid'];expected=s['process_identity']
    assert group==expected['group'] and members
    assert all(m['uid']==1013 and m['group']==group and m['session']==group for m in members)
    candidates=[m for m in members if m['python_executable'] and trainer in m['argv'] and '--output' in m['argv'] and m['argv'][m['argv'].index('--output')+1]==out]
    assert len(candidates)==1
    m=candidates[0];assert m['pid']==expected['pid'] and m['start_ticks']==expected['start_ticks']
    return m

def validate_backup(s,b,meta):
    assert b['status']=='FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED'
    assert b['run']==b['audit']['run']==RUN and b['completed']==b['audit']['completed_learning_iterations']==15000
    assert b['audit']['actual_optimizer_updates']==b['audit']['optimizer_step_min']==b['audit']['optimizer_step_max']==300000
    for key in ['private_full_two_location_sha_verified','inference_three_location_sha_verified','original_server_checkpoint_retained']:assert b[key] is True
    assert b['data_deleted'] is False and b['full_checkpoint_publicly_uploaded'] is False
    c=b['audit']['full_checkpoint'];assert c['path']==str(ROOT/'tracker_validation'/RUN/'completed_15000.pt')
    assert c['bytes']==meta['bytes'] and c['sha256']==meta['sha256']
    assert b['adapter']['spec_sha256']==s['backup_spec_sha256'] and b['adapter']['run']==RUN
    assert b['github_digest']=='sha256:'+b['audit']['public_inference']['sha256']

def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['observe','finish'],default='observe');a=p.parse_args()
    assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
    guard=(ROOT/'receipts/teacher15000_control.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sp=ROOT/'configs/teacher15000-direct-evaluation-spec.json';s=json.loads(sp.read_text())
    assert s['run']==RUN and s['evaluation_checkpoint']==15000 and s['paper_reference_iterations']==40000
    for n,h in s['bound_control_sources'].items():assert sha(ROOT/n)==h,n
    bp=ROOT/'configs/filter-recovery-backup-spec.json';assert sha(bp)==s['backup_spec_sha256'];bs=json.loads(bp.read_text())
    resume=ROOT/'configs/teacher10000-filter-recovery-spec.json';assert sha(resume)==s['resume_spec_sha256']
    rconf=json.loads(resume.read_text())
    for n,h in rconf['frozen_bindings'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'scripts/supervise_teacher10000_filter_recovery.py')==rconf['source_sha256']
    prefix='filter_recovery_'+RUN;out=ROOT/'tracker_validation'/RUN;rp=out/'receipt.json'
    pending=ROOT/'receipts/teacher15000_stop_requested.json';done=ROOT/'receipts/teacher15000_controlled_stop.json'
    backup_path=ROOT/'receipts/teacher15000_backup_verified.json'
    paths={k:ROOT/'receipts'/(prefix+suffix) for k,suffix in [('process','_process_exit.json'),('supervisor','_exit.json'),('outer','_outer_exit.json')]}
    def complete(req):
        assert all(x.exists() for x in paths.values()),'Await all real exits; never resend signals'
        validate_backup(s,json.loads(backup_path.read_text()),req['checkpoint'])
        assert sha(backup_path)==req['backup_receipt_sha256'] and sha(sp)==req['control_spec_sha256']
        ex={k:json.loads(v.read_text()) for k,v in paths.items()}
        assert ex['process']['child_pid']==req['owned_group'] and ex['process']['child_reaped']
        try: assert start_ticks(req['trainer_pid'])!=req['trainer_start_ticks'],'Original trainer still alive'
        except FileNotFoundError:pass
        assert sha(out/'completed_15000.pt')==req['checkpoint']['sha256']
        shared=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=dict(req,status='USER_AUTHORIZED_TEACHER15000_EVALUATION_HANDOFF',original_exits=ex,
          original_exit_sha256={k:sha(v) for k,v in paths.items()},final_training_receipt_sha256=sha(rp),
          common_sdk_lock_released=True,paper_40000_completed=False,physical_filter_completed=False,
          recoverable_completed=15000,recoverable_optimizer_updates=300000,transient_updates_after_checkpoint_not_guaranteed=True,timestamp_unix=time.time())
        with done.open('x') as f:json.dump(result,f,indent=2)
        return result
    if done.exists():print(done.read_text());return
    if pending.exists():
        req=json.loads(pending.read_text());assert req['run']==RUN
        if a.action=='observe':print(json.dumps(dict(req,status='TEACHER15000_STOP_PENDING',all_original_exits_observed=all(x.exists() for x in paths.values()))));return
        print(json.dumps(complete(req)));return
    assert not any(x.exists() for x in paths.values()),'Unrequested training exit requires inspection'
    launch=json.loads((ROOT/'receipts'/(prefix+'_launch.json')).read_text());live=json.loads((ROOT/'receipts'/(prefix+'_live.json')).read_text())
    members=[]
    for pr in pathlib.Path('/proc').iterdir():
        if not pr.name.isdigit():continue
        try:
            pid=int(pr.name)
            if os.getpgid(pid)!=live['child_pid']:continue
            members.append(dict(pid=pid,uid=pr.stat().st_uid,group=os.getpgid(pid),session=os.getsid(pid),start_ticks=start_ticks(pid),
                python_executable=(pr/'exe').resolve().name.startswith('python'),argv=(pr/'cmdline').read_bytes().decode(errors='replace').split('\0')))
        except (OSError,ProcessLookupError):continue
    identity=validate_identity(s,launch,live,members)
    check=preflight(ROOT,bs,15000)
    receipt=json.loads(rp.read_text())
    observation=dict(status='TEACHER15000_CHECKPOINT_READY' if check['ready'] else 'WAITING_VERIFIED_15000_CHECKPOINT',run=RUN,
        evaluation_checkpoint=15000,observed_completed=receipt['completed_learning_iterations'],optimizer_updates=receipt['optimizer_updates'],
        target_checkpoint_ready=check['ready'],checkpoint=check.get('metadata'),gate=check,trainer_pid=identity['pid'],trainer_start_ticks=identity['start_ticks'],
        owned_group=identity['group'],enabled=s['handoff_enabled'],signals_sent=[],sdk_runs=0,control_spec_sha256=sha(sp),sample_unix=time.time())
    if a.action=='observe':print(json.dumps(observation));return
    assert s['handoff_enabled'] and check['ready'] and not pending.exists()
    validate_backup(s,json.loads(backup_path.read_text()),check['metadata'])
    handle=BoundProcess(identity['pid'],identity['start_ticks'])
    assert os.getpgid(identity['pid'])==identity['group'] and pathlib.Path('/proc/'+str(identity['pid'])).stat().st_uid==1013
    req=dict(observation,action='user_authorized_pause_for_15000_physics_evaluation',backup_receipt_sha256=sha(backup_path),
        signal_identity='pidfd_original_process',user_request=s['user_request'],runtime_modified=False)
    with pending.open('x') as f:json.dump(req,f,indent=2)
    try:
        for sig,delay in [(signal.SIGINT,40),(signal.SIGTERM,15),(signal.SIGKILL,10)]:
            if all(x.exists() for x in paths.values()):break
            if handle.alive():
                handle.send(sig);req['signals_sent'].append(dict(pid=identity['pid'],signal=signal.Signals(sig).name,timestamp_unix=time.time()))
                tmp=pending.with_suffix('.partial');tmp.write_text(json.dumps(req,indent=2));tmp.replace(pending)
            deadline=time.monotonic()+delay
            while time.monotonic()<deadline and not all(x.exists() for x in paths.values()):time.sleep(.25)
    finally:handle.close()
    print(json.dumps(complete(req)))

if __name__=='__main__':main()
