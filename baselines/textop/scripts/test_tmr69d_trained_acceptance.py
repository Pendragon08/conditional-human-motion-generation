"""New isolated metadata corruptions; no real checkpoints, tensors or models."""
import copy
import json
import sys
from tmr69d_trained_acceptance import (TRAIN_RUN, TRAIN_SPEC, TRAIN_SPEC_SHA, TRAIN_ENTRY_SHA,
    TRAIN_RUNTIME_SHA, CHECKPOINT_DIR, validate_training_receipt, validate_fullstate_metadata)

spec = dict(run=TRAIN_RUN, formal_training_authorized=True, epochs=500, expected_optimizer_updates=507500,
    executor_sha256=TRAIN_ENTRY_SHA, runtime_binding_sha256=TRAIN_RUNTIME_SHA,
    accepted_input_signature_sha256='1'*64, diagnostic_initial_model_sha256='2'*64)
checkpoints = [dict(epoch=epoch, optimizer_updates=epoch*1015, path=str(CHECKPOINT_DIR/('tmr_epoch_'+str(epoch)+'.pth')),
    bytes=192900000, sha256='3'*64, pending_validation=False, emergency=False,
    exact_fullstate_deserialization=True, production_model_adamw_rng_sampler_nextbatch_reloaded=True) for epoch in [100,500]]
receipt = dict(run=TRAIN_RUN, status='TMR69D_FORMAL_EVALUATOR_500_EPOCHS_FINISHED', output_writers_closed=True,
    source_sha256=TRAIN_ENTRY_SHA, request_sha256=TRAIN_SPEC_SHA, formal_optimizer_updates=507500,
    attempted_optimizer_updates=507500, completed_epochs=500, skipped_updates=0,
    generator_or_teacher_training_updates=0, original_validation_sources_used_for_gradient=0,
    internal_validation_optimizer_updates=0, actual_original_validation_source_reads=0,
    exact_final_model_adamw_all_rng_sampler_nextbatch_disk_reload=True, model_sources_and_training_inputs_unchanged=True,
    original_class_method_AST_unchanged=True, inference_export_and_heldout_semantic_metrics_accepted=False,
    diagnostic_weights_used_for_initialization=False, actual_input_signature_sha256='1'*64,
    initial_model_sha256='4'*64, final_model_sha256='5'*64, checkpoints=checkpoints, selected_checkpoint=checkpoints[-1])
state = dict(optimizer_updates=507500, completed_epochs=500, pending_validation=False,
    request_path=str(TRAIN_SPEC), request_sha256=TRAIN_SPEC_SHA, runtime_binding_sha256=TRAIN_RUNTIME_SHA,
    input_signature_sha256='1'*64, sampler=dict(count=32462, cursor=32462, epoch=499, steps=507500),
    all_rng=dict(python=None,numpy=None,torch=None,cuda=[None]), next_batch_sha256='6'*64,
    model=dict(test=True), optimizer=dict(state={0:dict(step=507500)}))
assert validate_training_receipt(receipt,spec)==checkpoints[-1]
assert validate_fullstate_metadata(state,spec)['optimizer_updates']==507500
tests = []


def rejects(label, target, key, value):
    r,s,v = copy.deepcopy((receipt,spec,state))
    root = {'receipt':r,'spec':s,'state':v}[target]
    parts = key.split('.')
    for part in parts[:-1]:
        root = root[int(part)] if isinstance(root,list) else root[part]
    part = parts[-1]
    root[int(part) if isinstance(root,list) else part] = value
    try:
        validate_training_receipt(r,s)
        validate_fullstate_metadata(v,s)
    except (AssertionError,KeyError,ValueError,TypeError):
        tests.append(label)
    else:
        raise AssertionError('Corruption accepted: '+label)


for row in [
    ('still_running','receipt','status','RUNNING_FRESH_TMR69D_EVALUATOR_TRAINING'),
    ('writers_open','receipt','output_writers_closed',False),
    ('only_100_epochs','receipt','completed_epochs',100),
    ('one_update_missing','receipt','formal_optimizer_updates',507499),
    ('skip','receipt','skipped_updates',1),
    ('teacher_update','receipt','generator_or_teacher_training_updates',1),
    ('original_val_gradient','receipt','original_validation_sources_used_for_gradient',1),
    ('internal_val_updated','receipt','internal_validation_optimizer_updates',1),
    ('source_changed','receipt','source_sha256','7'*64),
    ('diagnostic_seed_weights','receipt','diagnostic_weights_used_for_initialization',True),
    ('no_production_restore','receipt','exact_final_model_adamw_all_rng_sampler_nextbatch_disk_reload',False),
    ('emergency_selected','receipt','checkpoints.1.emergency',True),
    ('pending_validation_receipt','receipt','checkpoints.1.pending_validation',True),
    ('bad_fullstate_path','receipt','checkpoints.1.path','/tmp/tmr_epoch_500.pth'),
    ('too_large_state','receipt','checkpoints.1.bytes',201326593),
    ('unbound_inputs','state','input_signature_sha256','7'*64),
    ('old_runtime','state','runtime_binding_sha256','7'*64),
    ('post_target_update','state','optimizer_updates',507501),
    ('pending_internal_validation','state','pending_validation',True),
    ('next_epoch_already_consumed','state','sampler.epoch',500),
    ('short_sampler','state','sampler.cursor',32461),
    ('no_cuda_rng_record','state','all_rng.cuda',[]),
    ('bad_next_batch_digest','state','next_batch_sha256','X'*64),
]:
    rejects(*row)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
print(json.dumps(dict(status='TMR_TRAINED_EXPORT_METADATA_CONTRACT_VERIFIED',isolated_negative_cases=len(tests),
    rejected=tests, actual_production_checkpoints_read=0, actual_model_loads=0, actual_gpu_contexts=0,
    actual_exports=0, actual_training_updates=0)))
