"""New isolated CPU fixture; the only signalled PID is this fixture's sleep.

No production checkpoint, trainer/observer or model is loaded or signalled.
Synthetic checkpoint bytes prove file/identity guards, not tensor correctness.
"""
import argparse,copy,json,os,pathlib,subprocess,sys,time
from tmr100_endpoint_protocol import checkpoint_ready,verify_checkpoint,sha,write_new,identity,verify_identity,open_bound_pidfd,send_once

def run(out):
    out=pathlib.Path(out);assert out.is_absolute() and out.parent.resolve()==out.parent
    assert out.parent.stat().st_uid==os.getuid()==1013;out.mkdir(mode=0o700,exist_ok=False)
    cp=out/'synthetic_epoch100.bin';cp.write_bytes(b'SYNTHETIC_CPU_ONLY_NOT_A_MODEL'*10)
    policy=dict(original_run='tmr69d_evaluator500_train_v1',original_spec_sha256='a'*64,
        original_entry_sha256='b'*64,target_epochs=100,target_optimizer_updates=101500,
        accepted_input_signature_sha256='c'*64,checkpoint_path=str(cp),checkpoint_upper_bytes=201326592)
    ck=dict(path=str(cp),epoch=100,optimizer_updates=101500,pending_validation=False,emergency=False,
        bytes=cp.stat().st_size,sha256=sha(cp),exact_fullstate_deserialization=True,
        production_model_adamw_rng_sampler_nextbatch_reloaded=True)
    rec=dict(run=policy['original_run'],request_sha256='a'*64,source_sha256='b'*64,
        status='RUNNING_FRESH_TMR69D_EVALUATOR_TRAINING',target_epochs=500,expected_optimizer_updates=507500,
        formal_optimizer_updates=101500,attempted_optimizer_updates=101500,skipped_updates=0,
        generator_or_teacher_training_updates=0,original_validation_sources_used_for_gradient=0,
        internal_validation_optimizer_updates=0,actual_input_signature_sha256='c'*64,
        fresh_model_and_adamw_initialized=True,diagnostic_weights_used_for_initialization=False,
        optimizer=dict(type='torch.optim.AdamW',lr=.0001,weight_decay=.01),completed_epochs=100,checkpoints=[ck])
    assert checkpoint_ready(rec,policy)==ck
    waiting=copy.deepcopy(rec);waiting.update(completed_epochs=99,formal_optimizer_updates=100485,attempted_optimizer_updates=100485,checkpoints=[])
    assert checkpoint_ready(waiting,policy) is None
    changes=[lambda x:x.update(request_sha256='d'*64),lambda x:x.update(source_sha256='e'*64),
        lambda x:x.update(status='FAILED_TMR69D_FORMAL_EVALUATOR_ALL_STATES_RETAINED'),
        lambda x:x.update(attempted_optimizer_updates=101501),lambda x:x.update(skipped_updates=1),
        lambda x:x.update(original_validation_sources_used_for_gradient=1),
        lambda x:x.update(actual_input_signature_sha256='d'*64),lambda x:x.update(diagnostic_weights_used_for_initialization=True),
        lambda x:x['checkpoints'][0].update(optimizer_updates=101499),lambda x:x['checkpoints'][0].update(pending_validation=True),
        lambda x:x['checkpoints'][0].update(emergency=True),
        lambda x:x['checkpoints'][0].update(production_model_adamw_rng_sampler_nextbatch_reloaded=False),
        lambda x:x['checkpoints'][0].update(path=str(out/'partial')),lambda x:x['checkpoints'].append(copy.deepcopy(ck))]
    for change in changes:
        bad=copy.deepcopy(rec);change(bad)
        try:checkpoint_ready(bad,policy)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Invalid target metadata accepted')
    proof=verify_checkpoint(ck,out)
    bad=dict(ck,sha256='0'*64)
    try:verify_checkpoint(bad,out)
    except AssertionError:pass
    else:raise AssertionError('Bad full SHA accepted')
    link=out/'synthetic_link';link.symlink_to(cp)
    try:verify_checkpoint(dict(ck,path=str(link)),out)
    except AssertionError:pass
    else:raise AssertionError('Symlink accepted')
    child=None;fd=None
    with (out/'sleep.log').open('xb') as stream:
        try:
            child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(60)'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            expected=identity(child.pid);assert any('time.sleep(60)' in x for x in expected['argv'])
            for bad in [dict(expected,start_ticks=str(int(expected['start_ticks'])+1)),dict(expected,argv=expected['argv']+['wrong'])]:
                try:verify_identity(bad)
                except AssertionError:pass
                else:raise AssertionError('Wrong process identity accepted')
            fd=open_bound_pidfd(expected)
            existing=out/'preexisting_intent.json';write_new(existing,dict(status='UNKNOWN_FIXTURE_INTENT_MUST_NOT_RETRY'))
            try:send_once(fd,expected,existing,dict(fixture_only=True))
            except FileExistsError:pass
            else:raise AssertionError('Duplicate/unknown intent was rearmed')
            assert child.poll() is None
            send_once(fd,expected,out/'fixture_signal_intent.json',dict(fixture_only=True,synthetic_checkpoint_proof=proof))
            code=child.wait(timeout=10);assert code!=0
            result=dict(status='TMR100_ENDPOINT_ISOLATED_CPU_FIXTURE_VERIFIED',metadata_negative_cases=len(changes),
                full_SHA_symlink_and_identity_negative_cases=4,unknown_duplicate_intent_rejected=True,
                actual_fixture_signals=1,actual_production_signals=0,fixture_child_exit_code=code,
                fixture_child_reaped=True,direct_streams_final=True,production_checkpoint_reads=0,
                production_trainer_process_signalled=False,model_loads=0,new_GPU_contexts=0,new_optimizer_updates=0)
        finally:
            if fd is not None:os.close(fd)
            if child is not None and child.poll() is None:
                child.terminate();child.wait(timeout=10)
            stream.flush();os.fsync(stream.fileno())
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    write_new(out/'receipt.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(a.out)))
