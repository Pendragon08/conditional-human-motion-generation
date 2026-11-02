"""Only actual closed100-epoch resumed training can feed frozen CPU export."""
import pathlib
from tmr100_capacity_resume_protocol import (ROOT, PARENT, HOME, BASE_SPEC, BASE_SHA,
    EMERGENCY_SHA, RUNS, owned, read, sha, origin, workers, closed_job, accepted_witnesses)

KIND = 'USER_TMR100_CAPACITY_RESUME_ORIGINAL_FAILURE_RETAINED'
REQUEST = ROOT / 'receipts/tmr100_capacity_resume_train_spec.json'


def validate_resume_provenance(state, request_sha256, runtime_binding_sha256):
    value = state['capacity_resume']
    assert value['request_path'] == str(REQUEST)
    assert value['request_sha256'] == request_sha256 and len(request_sha256) == 64
    assert value['runtime_binding_sha256'] == runtime_binding_sha256 and len(runtime_binding_sha256) == 64
    assert value['source_checkpoint_sha256'] == EMERGENCY_SHA
    assert value['original_safe_updates'] == 83703 and value['original_failed_attempts'] == 83704
    assert state['optimizer_updates'] == 101500 and state['completed_epochs'] == 100
    assert state['pending_validation'] is False


def accepted_resume(binding=None):
    actual = closed_job('train')
    spec, rec = actual['spec'], actual['receipt']
    assert actual['request_path'] == str(REQUEST)
    assert spec['formal_training_authorized'] is True and spec['witnesses_authorized'] is False
    base, source = origin(spec)
    document = spec['document_summary']
    dp = owned(document['path'])
    assert sha(dp) == document['sha256']
    witness = accepted_witnesses(read(dp))
    assert witness == spec['witness_bindings'] and witness['runtime_binding_sha256'] == spec['runtime_binding_sha256']
    assert not workers(), 'No old producer/controller or new resumed producer may remain'
    assert rec['run'] == RUNS['train'] and rec['status'] == 'TMR100_CAPACITY_RESUMED_100_EPOCHS_FINISHED'
    assert rec['formal_optimizer_updates'] == 101500 and rec['completed_epochs'] == 100
    assert rec['actual_new_formal_optimizer_updates'] == rec['attempted_new_optimizer_updates'] == 17797
    assert rec['skipped_updates'] == rec['generator_or_teacher_training_updates'] == rec['original_validation_sources_used_for_gradient'] == 0
    assert rec['internal_validation_optimizer_updates'] == 0 and rec['pending_validation'] is False
    assert rec['original500_success_claimed'] is False and rec['source_inputs_and_checkpoint_unchanged']
    assert rec['exact_final_model_adamw_all_rng_sampler_nextbatch_disk_reload']
    assert rec['selected_source_checkpoint'] == source
    assert rec['actual_input_signature_sha256'] == base['accepted_input_signature_sha256']
    selected = rec['selected_checkpoint']
    assert rec['checkpoints'] == [selected]
    cp = owned(selected['path'], HOME)
    assert cp.parent == pathlib.Path(spec['checkpoint_dir']) and cp.name == 'tmr_epoch_100_resumed.pth'
    assert cp.stat().st_size == selected['bytes'] <= 201326592 and cp.stat().st_nlink == 1
    assert sha(cp) == selected['sha256'] and selected['epoch'] == 100 and selected['optimizer_updates'] == 101500
    assert selected['exact_fullstate_deserialization'] and selected['production_model_adamw_rng_sampler_nextbatch_reloaded']
    assert selected['emergency'] is False and selected['pending_validation'] is False
    assert len(rec['final_model_sha256']) == 64
    result = dict(kind=KIND, run=RUNS['train'], selected_epochs=100, selected_optimizer_updates=101500,
        selected_checkpoint=selected, spec_sha256=actual['request_sha256'],
        runtime_binding_sha256=spec['runtime_binding_sha256'], source_emergency_sha256=EMERGENCY_SHA,
        source_safe_updates=83703, actual_new_formal_updates=17797,
        original_spec_sha256=BASE_SHA, original_failure_files=spec['original_failure_files'],
        resume_receipt_sha256=actual['sha256']['receipt'], closure=actual['sha256'],
        witness_bindings=witness, fresh_document_sha256=document['sha256'],
        input_signature=rec['actual_input_signature'], input_signature_sha256=rec['actual_input_signature_sha256'])
    if binding is not None:
        assert binding == result
    return dict(binding=result, spec=base, receipt=rec)
