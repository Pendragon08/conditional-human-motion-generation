"""Draft input binding for FK after the real interrupted-prefix integrity audit.

The caller must hold the model lock. No models or generated arrays are loaded
by this metadata adapter, apart from the two prior witness comparisons.
"""
import os

from filtered15000_vae_completion import ROOT, owned, read, sha
from augmentation5368_verified_inputs_v1 import accepted_inputs
from augmentation5368_witness_acceptance_v1 import accepted_witnesses
from augmentation2526_fk_plan_v1 import plan_fk
from augmentation2526_stop_binding_v1 import validate_interrupted_binding
from step4_train_only_protocol import scope

REPORT_SHA = '0bb020c4182227be0c119d305bd9d8f679c5fac9f9f801087667a375528d8394'
REQUEST_SHA = '419011b4909eea18dfe8eb7a698b8591d0b52d7261d5612875eea941c69ec806'
JOURNAL_SHA = '8d1b4997f6d4acb114b3db3fa74b9f04099d591cc9ffc3248e70d4e7951636cf'


def accepted_generated_fk_inputs(binding):
    assert os.getuid() == 1013
    assert binding == {'integrity_report_sha256': REPORT_SHA, 'production_request_sha256': REQUEST_SHA}
    parent = owned(ROOT / 'step34_20261029')
    report_path = owned(parent / 'user_augmentation2526_integrity_audit_v1/integrity.json')
    assert sha(report_path) == REPORT_SHA
    report = read(report_path)
    assert report['status'] == 'INTERRUPTED_PREFIX_INTEGRITY_PASSED'
    assert report['clips'] == 2526 and report['control_intervals'] == 2666420 and report['pose_frames'] == 2668946
    assert report['journal_sha256'] == JOURNAL_SHA and report['user_interrupted'] is True
    assert report['original_child_exit_code'] == -2 and report['original_outer_exit_code'] == 1
    assert report['final_in_memory_model_hash_verified'] is False
    assert report['model_loads'] == report['optimizer_updates'] == 0
    request = owned(ROOT / 'receipts/augmentation5368_user_augmentation5368_production_v1_request.json')
    assert sha(request) == REQUEST_SHA
    producer = read(request)
    producer_dir = owned(parent / 'user_augmentation5368_production_v1')
    log = owned(parent / 'user_augmentation5368_production_v1_supervision')
    paths = dict(request=request, receipt=producer_dir / 'receipt.json', exit=log / 'exit.json',
                 process=log / 'process_exit.json', resources=log / 'resources.jsonl',
                 outer=ROOT / 'receipts/augmentation5368_user_augmentation5368_production_v1_outer_exit.json')
    validate_interrupted_binding({key: sha(owned(path)) for key, path in paths.items()})
    assert report['closure_sha256'] == {key: sha(owned(path)) for key, path in paths.items()}
    assert sha(owned(producer_dir / 'clips.private.jsonl')) == JOURNAL_SHA
    assert sha(owned(ROOT / 'receipts/user_stop_augmentation5368_20261030_sent.json')) == 'd8868a98e35628933cc8e0867894563434039b71a9d5344e9b07bc634836da85'
    receipt = read(paths['receipt'])
    witnesses = accepted_witnesses(producer['witness_binding'])
    for witness in witnesses['witnesses']:
        assert witness['generator_model_sha256'] == receipt['generator_model_sha256']
        assert witness['vae_model_sha256'] == receipt['vae_model_sha256']
    actual = accepted_inputs()
    assert report['scope_data_sha256'] == actual['scope_data_sha256']
    _, sources, names, original_binding = scope()
    assert len(sources) == len(names) == 5361
    original_path = owned(ROOT / 'step34_numeric_v1/user_step4_train_only_fk_library_v1/receipt.json')
    original_sha = '375b16c3b8d919561e59cea8830ab071f2a128f39bf76363474cf0c37eb5ddde'
    assert sha(original_path) == original_sha
    original_receipt = read(original_path)
    assert original_receipt['status'] == 'STAGE4_ORIGINAL_TRAIN_ISAAC_REFERENCES_FINISHED'
    assert original_receipt['output_writers_closed'] is True and original_receipt['exported_clips'] == 5551
    assert len(original_receipt['joint_names']) == 29 and len(original_receipt['body_names']) == 30
    original_request_path = owned(ROOT / 'receipts/step4_train_only_user_step4_train_only_fk_library_v1_request.json')
    assert sha(original_request_path) == original_receipt['request_sha256'] == 'ce040884a55b7831cd47a6fa2e9323bb1d2ace00cd0b4ed2fc86b0725db9e752'
    original_request = read(original_request_path)
    audit_view = dict(status='TRAIN_AUGMENTATION_INTERRUPTED_PREFIX_2526_ARRAYS_AUDITED',
                      all_arrays_finite=True, all_quaternions_unit=True,
                      generated_clips=report['clips'], generated_train_items=report['items'])
    plan = plan_fk(audit_view, actual['plan'], names)
    assert plan['output_npz_upper_bytes'] == 5764264784
    return dict(binding=binding, plan=plan, original_train_binding=original_binding,
                original_library_receipt_sha256=original_sha,
                original_urdf_sha256=original_request['urdf_sha256'],
                original_joint_names=original_receipt['joint_names'], original_body_names=original_receipt['body_names'],
                witness_binding=producer['witness_binding'], witnesses=witnesses,
                final_in_memory_model_hash_verified=False, source_arrays_loaded=False,
                actual_fk=False, physical_tracking_accepted=False)
