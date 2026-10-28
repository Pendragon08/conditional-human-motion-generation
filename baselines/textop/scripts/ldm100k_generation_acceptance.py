"""Actual paired GPU inference witnesses, not an acceptance of text semantics."""
import hashlib
import json
from ldm100k_verified_input import ROOT, read, sha, owned, closed_job

RUNS = ['user_ldm100k_generation_author_v1', 'user_ldm100k_generation_document_v1']


def runtime_signature(spec):
    # Only authorization evidence can change between witnessed inference and
    # the full declared diagnostic. Data, numerics, budgets and sources freeze.
    value = {k: v for k, v in spec.items()
             if k not in ['witnesses_authorized', 'generation_authorized', 'witness_bindings']}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_witness(receipt, run, spec):
    r = receipt
    assert r['run'] == run and run in RUNS and r['mode'] == 'witness'
    assert r['status'] == 'USER_LDM100K_GENERATION_WITNESS_FINISHED' and r['output_writers_closed']
    assert r['source_sha256'] == spec['executor_sha256']
    assert r['planned_pairs'] == r['completed_pairs'] == 1
    assert r['planned_trials'] == r['completed_trials'] == 2
    assert r['failed_pairs'] == r['unattempted_pairs'] == 0
    assert r['training_optimizer_updates'] == 0 and r['actual_inference_model_loads'] == 2
    assert r['selected_ldm_attempts'] == 100000 and r['new_gpu_contexts'] == 1
    assert r['model_and_source_bytes_unchanged'] and r['verified_source_count'] == 1
    for key in ['future_posterior_encoded', 'semantic_metrics_computed',
                'physical_quality_accepted', 'original300k_completion_claimed',
                'full_validation_generation_claimed']:
        assert r[key] is False
    assert set(r['witness_pair_arrays']) == {'conditional_cfg5', 'unconditional'}
    assert len(r['witness_pair_probe_sha256']) == 64


def accepted_generation_witnesses(binding):
    sp = owned(ROOT / 'configs/ldm100k-generation-spec.json')
    s = read(sp)
    old = owned(ROOT / 'receipts/ldm100k_generation_spec_before_diagnostic.json')
    assert sha(old) == binding['witness_spec_sha256']
    ws = read(old)
    assert ws['witnesses_authorized'] and not ws['generation_authorized'] and ws['witness_bindings'] is None
    assert runtime_signature(ws) == runtime_signature(s) == binding['runtime_binding_sha256']
    records = []
    for run in RUNS:
        item = binding['runs'][run]
        rp = owned(ROOT / 'ldm100k_user_handoff_20261028' / run / 'receipt.json')
        assert sha(rp) == item['receipt_sha256']
        rec = read(rp)
        validate_witness(rec, run, ws)
        assert rec['spec_sha256'] == sha(old)
        closure = closed_job('ldm100k_generation_' + run, rp, sha(rp), ws['supervisor_sha256'])
        assert closure == item['closure'] and closure['exit_sha256'] == item['exit_sha256']
        records.append(rec)
    for key in ['scope_receipt_sha256', 'trained_denoiser_sha256', 'frozen_vae_sha256',
                'witness_pair_arrays', 'witness_pair_probe_sha256']:
        assert records[0][key] == records[1][key]
    dp = owned(binding['document_path'])
    assert sha(dp) == binding['document_sha256']
    doc = read(dp)
    assert doc['status'] == 'USER_LDM100K_GENERATION_FRESH_DOCUMENT_VERIFIED'
    assert doc['run'] == RUNS[1] and type(doc['actual_exit_code']) is int and doc['actual_exit_code'] == 0
    assert doc['findings'] == [] and doc['source_sha256'] == ws['executor_sha256']
    assert doc['runtime_binding_sha256'] == binding['runtime_binding_sha256']
    assert doc['actual_witness_receipt_sha256'] == binding['runs'][RUNS[1]]['receipt_sha256']
    assert doc['actual_witness_pair_arrays'] == records[1]['witness_pair_arrays']
    assert doc['actual_witness_pair_probe_sha256'] == records[1]['witness_pair_probe_sha256']
    return dict(records=records, runtime_binding_sha256=binding['runtime_binding_sha256'])
