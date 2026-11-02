"""Immutable-byte contract for the accepted resumed100-epoch evaluator only."""
PATH = '/home/ouyanghao/textop-tmr69d-evaluator-20261029/resume_checkpoints_v1/tmr_epoch_100_resumed.pth'
KIND = 'USER_TMR100_CAPACITY_RESUME_ORIGINAL_FAILURE_RETAINED'


def validate_descriptor(checkpoint, export_binding, training_binding):
    c, e, t = checkpoint, export_binding, training_binding
    assert c['path'] == PATH
    assert type(c['bytes']) is int and 0 < c['bytes'] <= 201326592
    for value in [c['epoch'], c['optimizer_updates'], t['selected_epochs'],
                  t['selected_optimizer_updates'], t['source_safe_updates'], t['actual_new_formal_updates']]:
        assert type(value) is int
    assert c['epoch'] == 100 and c['optimizer_updates'] == 101500
    assert c['pending_validation'] is False and c['emergency'] is False
    assert c['exact_fullstate_deserialization'] is True
    assert c['production_model_adamw_rng_sampler_nextbatch_reloaded'] is True
    assert t['kind'] == KIND and t['run'] == 'tmr100_capacity_resume_train_v1'
    assert t['selected_epochs'] == 100 and t['selected_optimizer_updates'] == 101500
    assert t['source_safe_updates'] == 83703 and t['actual_new_formal_updates'] == 17797
    assert t['selected_checkpoint'] == c
    assert e['run'] == 'tmr100_frozen_export_v1'
    assert set(e['closure']) == {'exit_sha256', 'process_sha256', 'resources_sha256', 'outer_sha256'}
    values = [c['sha256'], e['spec_sha256'], e['receipt_sha256'],
              t['spec_sha256'], t['runtime_binding_sha256'], t['resume_receipt_sha256'],
              t['fresh_document_sha256']] + list(e['closure'].values())
    for value in values:
        assert isinstance(value, str) and len(value) == 64
        assert all(x in '0123456789abcdef' for x in value)
