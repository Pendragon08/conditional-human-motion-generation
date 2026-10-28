"""Read-only actual CPU export closure/provenance; no model imports."""
from ldm100k_prefix_completion import ROOT, read, sha, owned, closed_job, stopped_ldm100k, verify_checkpoint


def validate_export_receipt(receipt, spec, binding, checkpoint, control_receipt_sha):
    assert receipt['run'] == binding['run'] == 'user_ldm100k_prefix_export_v1'
    assert receipt['status'] == 'USER_LDM100000_PREFIX_CPU_EXPORT_PASSED'
    assert receipt['output_writers_closed'] and receipt['exact_inference_reload']
    assert receipt['source_sha256'] == spec['executor_sha256']
    assert receipt['spec_sha256'] == binding['spec_sha256']
    assert receipt['control_receipt_sha256'] == control_receipt_sha
    assert receipt['selected_checkpoint'] == checkpoint
    assert receipt['inference'] == binding['inference']
    assert receipt['source_full_state_preserved'] and receipt['original_interrupted_exit_retained']
    assert receipt['new_gpu_contexts'] == receipt['new_sdk_tasks'] == 0
    assert receipt['training_optimizer_updates_performed'] == 0
    assert receipt['original300k_completion_claimed'] is False
    assert receipt['full_native_schedule_completed'] is False
    assert receipt['semantic_quality_accepted'] is False
    assert receipt['cuda_rng_restored'] is False
    assert receipt['cpu_full_state_loads'] == 1 and receipt['cpu_model_loads'] == 2
    assert receipt['adam_step_min'] == receipt['adam_step_max'] == checkpoint['actual_optimizer_updates']
    assert receipt['finite_cpu_tensors'] > 0
    probe = receipt['synthetic_cpu_reload_probe']
    assert probe == dict(batch=2, conditional_and_unconditional=True, forward_calls=4,
                         exact=True, heldout_semantics=False)


def accepted_export(binding):
    sp = owned(ROOT / 'configs/ldm100k-prefix-export-spec.json')
    assert sha(sp) == binding['spec_sha256']
    spec = read(sp)
    assert spec['export_authorized'] and spec['ldm_training']
    assert sha(ROOT / 'scripts/export_ldm100k_prefix.py') == spec['executor_sha256']
    assert sha(ROOT / 'scripts/supervise_ldm100k_prefix_export.py') == spec['supervisor_sha256']
    for n, h in spec['bound_sources'].items():
        assert sha(ROOT / n) == h, n
    v = stopped_ldm100k(spec['ldm_training'])
    rp = owned(ROOT / 'ldm100k_user_handoff_20261028' / binding['run'] / 'receipt.json')
    assert sha(rp) == binding['receipt_sha256']
    receipt = read(rp)
    validate_export_receipt(receipt, spec, binding, v['checkpoint'], v['control_receipt_sha256'])
    proof = closed_job('ldm100k_prefix_export_' + binding['run'], rp, sha(rp), spec['supervisor_sha256'])
    assert proof['exit_sha256'] == binding['exit_sha256']
    artifact = receipt['inference']
    assert artifact['path'] == spec['export_dir'] + '/ldm100000_inference.private.pth'
    assert 0 < artifact['bytes'] <= spec['export_upper_bound_bytes']
    # An inference artifact is a regular owned file too; the shared checker
    # deliberately checks immutable bytes rather than deserializing a model.
    verify_checkpoint(artifact, spec['export_allowed_root'])
    return dict(stopped=v, export_spec=spec, receipt=receipt, closure=proof,
                inference=artifact, receipt_sha256=sha(rp))
