"""Actual closed independent CPU verifier plus real fresh-document evidence.

Standard library only; never constructs a model or loads a tensor/array.
"""
from ldm100k_export_acceptance import ROOT, read, sha, owned, closed_job, accepted_export


def validate_document(summary, receipt, binding):
    assert summary['status'] == 'USER_LDM100K_PREFIX_FRESH_DOCUMENT_VERIFIED'
    assert summary['run'] == receipt['run'] == binding['run'] == 'user_ldm100k_prefix_document_v1'
    assert type(summary['actual_exit_code']) is int and summary['actual_exit_code'] == 0
    assert summary['findings'] == []
    for k, rk in [('executor_sha256', 'source_sha256'), ('spec_sha256', 'spec_sha256'),
                  ('export_receipt_sha256', 'export_receipt_sha256')]:
        assert summary[k] == receipt[rk]
    assert summary['selected_checkpoint_sha256'] == receipt['selected_checkpoint']['sha256']
    assert summary['inference_sha256'] == receipt['inference']['sha256']
    assert summary['actual_metadata_audit'] == receipt['metadata_audit']
    assert summary['actual_closure'] == binding['closure']


def accepted_verified_input(binding):
    assert binding['run'] == 'user_ldm100k_prefix_document_v1'
    sp = owned(ROOT / 'configs/ldm100k-prefix-verify-spec.json')
    assert sha(sp) == binding['spec_sha256']
    s = read(sp)
    assert s['verify_authorized'] and s['export_binding']
    assert sha(ROOT / 'scripts/verify_ldm100k_prefix_export.py') == s['executor_sha256']
    assert sha(ROOT / 'scripts/supervise_ldm100k_prefix_verify.py') == s['supervisor_sha256']
    for n, h in s['bound_sources'].items():
        assert sha(ROOT / n) == h, n
    rp = owned(ROOT / 'ldm100k_user_handoff_20261028' / binding['run'] / 'receipt.json')
    assert sha(rp) == binding['receipt_sha256']
    rec = read(rp)
    assert rec['status'] == 'USER_LDM100K_PREFIX_FRESH_CPU_VERIFIED' and rec['output_writers_closed']
    assert rec['source_sha256'] == s['executor_sha256'] and rec['spec_sha256'] == sha(sp)
    assert rec['fullstate_and_exported_denoiser_exact'] and rec['cpu_adam_restore_exact']
    assert rec['synthetic_cpu_forward_exact'] and rec['source_and_inference_fingerprints_unchanged']
    assert rec['training_optimizer_updates_performed'] == rec['new_gpu_contexts'] == rec['new_sdk_tasks'] == 0
    assert rec['original300k_completion_claimed'] is False and rec['semantic_quality_accepted'] is False
    proof = closed_job('ldm100k_prefix_verify_' + binding['run'], rp, sha(rp), s['supervisor_sha256'])
    assert proof == binding['closure'] and proof['exit_sha256'] == binding['exit_sha256']
    export = accepted_export(s['export_binding'])
    assert rec['selected_checkpoint'] == export['stopped']['checkpoint']
    assert rec['inference'] == export['inference'] and rec['export_receipt_sha256'] == export['receipt_sha256']
    assert rec['trained_denoiser_sha256'] == export['receipt']['trained_denoiser_sha256']
    dp = owned(binding['document_path'])
    assert sha(dp) == binding['document_sha256']
    validate_document(read(dp), rec, binding)
    return dict(receipt=rec, receipt_sha256=sha(rp), closure=proof, export=export,
                document_sha256=sha(dp), document_path=str(dp))
