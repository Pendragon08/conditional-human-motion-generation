"""Standard-library100-epoch endpoint; original500-epoch source stays frozen."""
from vae100k_control_protocol import (read,sha,write_new,verify_checkpoint,identity,
    verify_identity,open_bound_pidfd,send_once)

RUN='tmr69d_evaluator500_train_v1'
CONTROL_RUN='user_tmr100_control_v1'
TARGET_EPOCHS=100
TARGET_UPDATES=101500


def checkpoint_ready(rec,policy):
    assert rec['run']==policy['original_run']==RUN
    assert rec['request_sha256']==policy['original_spec_sha256']
    assert rec['source_sha256']==policy['original_entry_sha256']
    assert rec['status']=='RUNNING_FRESH_TMR69D_EVALUATOR_TRAINING'
    assert rec['target_epochs']==500 and rec['expected_optimizer_updates']==507500
    assert policy['target_epochs']==TARGET_EPOCHS and policy['target_optimizer_updates']==TARGET_UPDATES
    assert rec['formal_optimizer_updates']==rec['attempted_optimizer_updates'] and rec['skipped_updates']==0
    assert rec['generator_or_teacher_training_updates']==rec['original_validation_sources_used_for_gradient']==0
    assert rec['internal_validation_optimizer_updates']==0
    assert rec['actual_input_signature_sha256']==policy['accepted_input_signature_sha256']
    assert rec['fresh_model_and_adamw_initialized'] and not rec['diagnostic_weights_used_for_initialization']
    assert rec['optimizer']==dict(type='torch.optim.AdamW',lr=.0001,weight_decay=.01)
    assert type(rec['completed_epochs']) is int and type(rec['formal_optimizer_updates']) is int
    assert 1015*rec['completed_epochs']<=rec['formal_optimizer_updates']
    matches=[x for x in rec['checkpoints'] if x['epoch']==TARGET_EPOCHS]
    if not matches:return None
    assert len(matches)==1
    ck=matches[0]
    assert ck['path']==policy['checkpoint_path'] and ck['optimizer_updates']==TARGET_UPDATES
    assert rec['completed_epochs']>=TARGET_EPOCHS and rec['formal_optimizer_updates']>=TARGET_UPDATES
    assert ck['pending_validation'] is False and ck['emergency'] is False
    assert ck['exact_fullstate_deserialization'] is True
    assert ck['production_model_adamw_rng_sampler_nextbatch_reloaded'] is True
    assert type(ck['bytes']) is int and 0<ck['bytes']<=policy['checkpoint_upper_bytes']==201326592
    assert len(ck['sha256'])==64 and all(x in '0123456789abcdef' for x in ck['sha256'])
    return ck
