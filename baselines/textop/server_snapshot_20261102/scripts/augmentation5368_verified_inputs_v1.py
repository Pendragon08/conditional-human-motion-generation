"""Read-only provenance gate for the authorized TRAIN augmentation plan.

This module does not import a numerical framework, read motion arrays or start
a model. The caller must independently hold the shared model lock for all
numerical work and must verify the gate's own source hash.
"""
import json

from filtered15000_vae_completion import ROOT, owned, read, sha
from ldm100k_verified_input import accepted_verified_input
from augmentation5368_protocol import validate_plan

SCOPE_RUN = 'user_augmentation5368_scope_v1'
SCOPE_SPEC = 'e0b5086ef7547d6bffc7161cf715f19b427640c63638599b7b83b65021d030ed'
SCOPE_RECEIPT = 'ca9f6d2c6f6dd15e8f2c46987b1e8094d0fd94e14977e393201a9aa454cce901'
SCOPE_DATA = 'ac49c12afeedc8703cf2c3b8c8e19df4c25fc691c2220df9d927d32d3552cbb4'
TRAIN_RECEIPT = 'f355de1c947487d750cbe3d59c51d78c8853a36315c659a3b08b4876df7273d2'
TRAIN_DATA = 'c88f73125f16b959b84eb2f9d33f4c17b40287c3b8b9a058047035070df0450a'
MODEL_SCOPE_SPEC = '8ca7d37dc806a281c67be76380f223c9a55607ddf10e777078b8c9f640897ff9'


def accepted_inputs():
    """Return actual TRAIN-only sources/plan and a verified frozen generator."""
    folder = owned(ROOT / 'step34_20261029' / SCOPE_RUN)
    log = owned(ROOT / 'step34_20261029' / (SCOPE_RUN + '_supervision'))
    spec_path = owned(ROOT / 'configs/user-augmentation5368-scope-spec.json')
    assert sha(spec_path) == SCOPE_SPEC
    spec = read(spec_path)
    assert spec['run'] == SCOPE_RUN and spec['scope_authorized'] is True
    assert spec['generation_authorized'] is False and spec['ppo_train_authorized'] is False
    for name, digest in spec['bound_sources'].items():
        assert sha(owned(ROOT / name)) == digest
    rp = owned(folder / 'receipt.json')
    assert sha(rp) == SCOPE_RECEIPT
    rec = read(rp)
    assert rec['status'] == 'USER_STEP3_AUGMENTATION5368_METADATA_SCOPE_BUILT'
    assert rec['output_writers_closed'] is True and rec['request_sha256'] == SCOPE_SPEC
    ex = read(owned(log / 'exit.json'))
    proc = read(owned(log / 'process_exit.json'))
    outer = read(owned(ROOT / 'receipts' / ('step34_' + SCOPE_RUN + '_outer_exit.json')))
    for proof in (ex, proc):
        assert proof['supervision_passed'] is True and proof['child_exit_code'] == 0
        assert proof['child_reaped'] is True and proof['timed_out'] is False and proof['sample_errors'] == []
    assert outer['actual_returncode'] == 0 and ex['receipt_sha256'] == SCOPE_RECEIPT
    assert ex['request_sha256'] == SCOPE_SPEC and ex['source_sha256'] == spec['supervisor_sha256']
    assert ex['process_exit_sha256'] == sha(owned(log / 'process_exit.json'))
    resources = owned(log / 'resources.jsonl')
    assert ex['resources_sha256'] == sha(resources)
    lines = resources.read_text().splitlines()
    assert len(lines) == ex['samples_written']
    last = json.loads(lines[-1])
    assert last['phase'] == 'after_exit' and last['owned_group_processes'] == last['owned_group_gpu_memory'] == []
    item = rec['private_artifact']
    plan_path = owned(item['path'])
    assert plan_path == folder / 'augmentation_scope.private.json'
    assert plan_path.stat().st_size == item['bytes'] and sha(plan_path) == item['sha256'] == SCOPE_DATA
    train_receipt_path = owned(spec['train_scope_receipt']['path'])
    assert sha(train_receipt_path) == spec['train_scope_receipt']['sha256'] == TRAIN_RECEIPT
    train_receipt = read(train_receipt_path)
    assert train_receipt['status'] == 'TMR69D_TRAINING_METADATA_SCOPE_BUILT' and train_receipt['output_writers_closed'] is True
    train_item = train_receipt['private_artifact']
    train_path = owned(train_item['path'])
    assert train_path.stat().st_size == train_item['bytes'] and sha(train_path) == train_item['sha256'] == TRAIN_DATA
    train = read(train_path)
    source_ids = {x['canonical_source'] for x in train['sources']}
    assert len(source_ids) == len(train['sources']) == 5361
    assert all(x['original_split'] == 'train' and x['mirrored'] is False for x in train['sources'])
    plan = read(plan_path)
    assert plan['schema'] == 'textop.user_aug5368_private_scope.v1'
    validate_plan(plan['clips'], source_ids, set(train['texts']))
    assert plan['summary']['actual_generated_clips'] == plan['summary']['actual_RL_updates'] == 0
    model_spec = owned(ROOT / 'configs/ldm100k-heldout-scope-spec.json')
    assert sha(model_spec) == MODEL_SCOPE_SPEC
    verified = accepted_verified_input(read(model_spec)['verified_model'])
    return dict(plan=plan, train_index=train['sources'], verified=verified,
                scope_receipt_sha256=SCOPE_RECEIPT, scope_data_sha256=SCOPE_DATA,
                train_receipt_sha256=TRAIN_RECEIPT, train_data_sha256=TRAIN_DATA)
