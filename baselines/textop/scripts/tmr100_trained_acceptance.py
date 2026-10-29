"""Genuine100-epoch prefix provenance, retaining interrupted500-epoch exits."""
import json,os,pathlib
from filtered15000_vae_completion import ROOT,read,sha,owned
from tmr100_endpoint_protocol import checkpoint_ready,verify_checkpoint
from tmr69d_trained_acceptance import digest,fingerprint
HOME=pathlib.Path('/home/ouyanghao')
PARENT=ROOT/'evaluator_tmr_20261029'
TRAIN_RUN='tmr69d_evaluator500_train_v1'
TRAIN_SPEC=ROOT/'configs/tmr69d-evaluator-train-spec.json'
TRAIN_SPEC_SHA='2196a765dbaa61ba7265b9d798f99fe5c094a502fd34bd47869f0fe141360b13'
TRAIN_ENTRY_SHA='04014d8f2e657e9c48827edd7fae817c3f01362d97dcdbd3a0e3dea092c604c2'
TRAIN_RUNTIME_SHA='c89a006b8730db6f767e0e3542506a3b9be928aaa12679f2ae6a1b0e8c4b7030'
CHECKPOINT_DIR=HOME/'textop-tmr69d-evaluator-20261029/checkpoints'


def validate_fullstate_metadata(state,spec):
    assert state['optimizer_updates']==101500 and state['completed_epochs']==100
    assert state['pending_validation'] is False
    assert state['request_path']==str(TRAIN_SPEC) and state['request_sha256']==TRAIN_SPEC_SHA
    assert state['runtime_binding_sha256']==TRAIN_RUNTIME_SHA
    assert state['input_signature_sha256']==spec['accepted_input_signature_sha256']
    sampler=state['sampler']
    assert sampler['count']==sampler['cursor']==32462
    assert sampler['epoch']==99 and sampler['steps']==101500
    assert set(state['all_rng'])=={'python','numpy','torch','cuda'}
    assert isinstance(state['all_rng']['cuda'],list) and len(state['all_rng']['cuda'])==1
    digest(state['next_batch_sha256'])
    assert isinstance(state['model'],dict) and state['model']
    assert isinstance(state['optimizer'],dict) and state['optimizer']['state']
    if 'capacity_resume' in state:
        from tmr100_capacity_resumed_acceptance import accepted_resume,validate_resume_provenance
        resumed=accepted_resume()
        validate_resume_provenance(state,resumed['binding']['spec_sha256'],resumed['binding']['runtime_binding_sha256'])
    return dict(completed_epochs=100,optimizer_updates=101500,sampler_epoch=99,sampler_cursor=32462,
        pending_validation=False,saved_cuda_rng_devices=1,cuda_rng_restored=False,actual_new_optimizer_updates=0,
        selected_user_prefix=True,original500_epoch_success_claimed=False)


def closed_evaluator_job(run,receipt_path,receipt_sha,supervisor_sha,request_sha,training=False):
    assert not training,'Original interrupted training uses the separate100-prefix controller closure'
    assert run in ['tmr100_frozen_export_v1','tmr100_frozen_verify_v1']
    rp=owned(receipt_path);assert rp==PARENT/run/'receipt.json' and sha(rp)==receipt_sha
    log=owned(PARENT/(run+'_supervision'));prefix='tmr100_frozen_cpu_'+run
    paths=dict(exit=log/'exit.json',process=log/'process_exit.json',resources=log/'resources.jsonl',
        outer=ROOT/'receipts'/(prefix+'_outer_exit.json'))
    paths={k:owned(p) for k,p in paths.items()}
    ex,pr,outer=[read(paths[k]) for k in ['exit','process','outer']]
    assert outer['actual_returncode']==0
    for proof in [ex,pr]:
        assert proof['supervision_passed'] and proof['child_reaped'] and proof['child_exit_code']==0
        assert not proof['timed_out'] and proof['supervisor_error'] is None and proof['sample_errors']==[]
        assert proof['shutdown_actions']==[] and not proof.get('cleanup_error')
    assert ex['receipt_sha256']==receipt_sha and ex['source_sha256']==supervisor_sha and ex['request_sha256']==request_sha
    assert ex['process_exit_sha256']==sha(paths['process']) and ex['resource_sha256']==sha(paths['resources'])
    rows=[json.loads(x) for x in paths['resources'].read_text().splitlines()]
    assert len(rows)==ex['samples_written']==pr['samples_written'] and len(rows)>=3
    assert rows[0]['phase']=='before_launch' and rows[-1]['phase']=='after_exit' and rows[-1]['owned_group_processes']==[]
    return {name+'_sha256':sha(path) for name,path in paths.items()}


def accepted_training(binding=None):
    if (binding is not None and binding.get('kind')=='USER_TMR100_CAPACITY_RESUME_ORIGINAL_FAILURE_RETAINED') or (binding is None and (PARENT/'tmr100_capacity_resume_train_v1/receipt.json').exists()):
        from tmr100_capacity_resumed_acceptance import accepted_resume
        return accepted_resume(binding)
    sp=owned(TRAIN_SPEC);assert sha(sp)==TRAIN_SPEC_SHA;spec=read(sp)
    assert sha(ROOT/'scripts/train_tmr69d_evaluator.py')==spec['executor_sha256']==TRAIN_ENTRY_SHA
    assert sha(ROOT/'scripts/supervise_tmr69d_evaluator_training.py')==spec['supervisor_sha256']
    assert spec['epochs']==500 and spec['expected_optimizer_updates']==507500 and spec['save_epochs']==[100,500]
    assert spec['runtime_binding_sha256']==TRAIN_RUNTIME_SHA
    for name,value in spec['bound_sources'].items():assert sha(owned(ROOT/name))==value
    for name in ['scope_receipt','numeric_receipt','text_receipt','import_receipt']:
        item=spec['inputs'][name];path=owned(item['path'])
        assert sha(path)==item['sha256'] and read(path)['output_writers_closed']
    for name,value in spec['upstream_model_sha256'].items():assert sha(owned(pathlib.Path(spec['upstream_model_dir'])/(name+'.py')))==value
    policy_path=owned(ROOT/'configs/tmr100-user-control-spec.json');policy=read(policy_path)
    assert policy['control_authorized'] and policy['original_spec_sha256']==TRAIN_SPEC_SHA
    assert policy['original_runtime_binding_sha256']==TRAIN_RUNTIME_SHA
    assert sha(ROOT/'scripts/watch_tmr100_endpoint.py')==policy['entrypoint_sha256']
    assert sha(ROOT/'scripts/supervise_tmr100_endpoint.py')==policy['supervisor_sha256']
    for name,value in policy['bound_sources'].items():assert sha(owned(ROOT/name))==value
    folder=owned(PARENT/'user_tmr100_control_v1');rp=owned(folder/'receipt.json');rec=read(rp)
    assert rec['status']=='USER_TMR100_EPOCH_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED'
    assert rec['output_writers_closed'] and rec['signals_sent']==1 and rec['common_model_lock_released']
    assert rec['selected_epochs']==100 and rec['selected_optimizer_updates']==101500
    assert not rec['original500_completion_claimed'] and not rec['original_training_source_modified']
    assert rec['source_sha256']==policy['entrypoint_sha256'] and rec['spec_sha256']==sha(policy_path)
    before=owned(folder/'producer_before_signal.json');snapshot=read(before)
    ck=checkpoint_ready(snapshot,policy);assert ck==rec['selected_checkpoint']
    prefix='tmr100_control_user_tmr100_control_v1'
    paths=dict(exit=ROOT/'receipts'/(prefix+'_exit.json'),process=ROOT/'receipts'/(prefix+'_process_exit.json'),
        outer=ROOT/'receipts'/(prefix+'_outer_exit.json'),resources=ROOT/'logs'/(prefix+'_resources.jsonl'))
    paths={k:owned(p) for k,p in paths.items()};ex,pr,outer=[read(paths[k]) for k in ['exit','process','outer']]
    assert outer['actual_returncode']==0
    for value in [ex,pr]:
        assert value['child_reaped'] and value['child_exit_code']==0 and value['supervision_passed']
        assert not value['timed_out'] and value['sample_errors']==[] and value['supervisor_error'] is None
        assert value['shutdown_actions']==[] and not value.get('cleanup_error')
    assert ex['receipt_sha256']==sha(rp) and ex['request_sha256']==sha(policy_path) and ex['source_sha256']==policy['supervisor_sha256']
    assert ex['process_exit_sha256']==sha(paths['process']) and ex['resource_sha256']==sha(paths['resources'])
    rows=[json.loads(x) for x in paths['resources'].read_text().splitlines()]
    assert len(rows)==ex['samples_written']==pr['samples_written'] and rows[-1]['phase']=='after_exit' and rows[-1]['owned_group_processes']==[]
    original_log=owned(PARENT/(TRAIN_RUN+'_supervision'))
    original_paths=dict(exit=original_log/'exit.json',process=original_log/'process_exit.json',
        outer=ROOT/'receipts'/('tmr_evaluator_train_'+TRAIN_RUN+'_outer_exit.json'))
    assert {k:sha(owned(p)) for k,p in original_paths.items()}==rec['original_exit_sha256']
    final_path=owned(PARENT/TRAIN_RUN/'receipt.json');final=read(final_path)
    assert sha(final_path)==rec['original_final_receipt_sha256']
    assert final['status']=='FAILED_TMR69D_FORMAL_EVALUATOR_ALL_STATES_RETAINED'
    assert final['error']=="RuntimeError('Owned TMR evaluator interrupted with signal 2')"
    assert final['request_sha256']==TRAIN_SPEC_SHA and final['source_sha256']==TRAIN_ENTRY_SHA
    assert read(original_paths['outer'])['actual_returncode']!=0
    for name in ['exit','process']:
        value=read(original_paths[name]);assert value['child_reaped'] and value['child_exit_code']!=0
        assert not value['timed_out'] and value['supervisor_error'] is None and value['sample_errors']==[]
    assert sha(original_log/'resources.jsonl')==rec['original_resources_sha256']
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=1013:continue
            argv=(proc/'cmdline').read_bytes().decode().split('\0')
            assert not any(str(ROOT/'scripts'/name) in argv for name in ['train_tmr69d_evaluator.py','supervise_tmr69d_evaluator_training.py','watch_tmr100_endpoint.py','supervise_tmr100_endpoint.py'])
        except (FileNotFoundError,ProcessLookupError):pass
    proof=verify_checkpoint(ck,'/home/ouyanghao')
    result=dict(kind='USER_TMR100_EPOCH_PREFIX_ORIGINAL_EXITS_RETAINED',run=TRAIN_RUN,
        selected_epochs=100,selected_optimizer_updates=101500,selected_checkpoint=ck,
        spec_sha256=TRAIN_SPEC_SHA,runtime_binding_sha256=TRAIN_RUNTIME_SHA,
        control_receipt_sha256=sha(rp),control_spec_sha256=sha(policy_path),
        closure={k+'_sha256':sha(p) for k,p in paths.items()},
        producer_before_signal_sha256=sha(before),original_final_receipt_sha256=sha(final_path),
        input_signature=snapshot['actual_input_signature'],input_signature_sha256=snapshot['actual_input_signature_sha256'])
    if binding is not None:assert binding==result
    return dict(binding=result,spec=spec,receipt=rec)


def accepted_frozen_export(binding):
    sp=owned(ROOT/'configs/tmr100-frozen-export-spec.json');assert sha(sp)==binding['spec_sha256']
    spec=read(sp);assert spec['authorized'] and spec['role']=='export'
    assert spec['run']==binding['run']=='tmr100_frozen_export_v1'
    assert sha(ROOT/'scripts/run_tmr100_frozen_cpu.py')==spec['executor_sha256']
    assert sha(ROOT/'scripts/supervise_tmr100_frozen_cpu.py')==spec['supervisor_sha256']
    for name,value in spec['bound_sources'].items():assert sha(owned(ROOT/name))==value
    train=accepted_training(spec['training_binding']);rp=owned(PARENT/spec['run']/'receipt.json')
    assert sha(rp)==binding['receipt_sha256'];rec=read(rp)
    assert rec['status']=='TMR100_TRAINED_FROZEN_CPU_EXPORTED' and rec['output_writers_closed']
    assert rec['training_binding']==train['binding'] and rec['spec_sha256']==sha(sp) and rec['source_sha256']==spec['executor_sha256']
    for key in ['full_model_and_cpu_adamw_restored_exact','independent_inference_reload_exact','synthetic_mean_forward_exact','original_fullstate_unchanged']:assert rec[key] is True
    assert rec['training_optimizer_updates_performed']==rec['new_cuda_contexts']==0 and rec['cuda_rng_restored'] is False and rec['semantic_metrics_accepted'] is False
    digest(rec['trained_model_sha256']);assert rec['fullstate_metadata_audit']['optimizer_updates']==101500
    assert rec['optimizer_step_min']==rec['optimizer_step_max']==101500
    item=rec['inference'];assert item['path']==spec['inference_path'] and 0<item['bytes']<=spec['inference_upper_bytes']
    fp=fingerprint(item['path'],home=True);assert {k:fp[k] for k in ['path','bytes','sha256']}==item
    closure=closed_evaluator_job(spec['run'],rp,sha(rp),spec['supervisor_sha256'],sha(sp))
    assert binding==dict(run=spec['run'],spec_sha256=sha(sp),receipt_sha256=sha(rp),closure=closure,inference=item)
    return dict(binding=binding,receipt=rec,spec=spec,training=train)
