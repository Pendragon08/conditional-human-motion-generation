"""New isolated metadata contract fixtures, no real files/models or dispatch."""
import copy
import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from ldm100k_scope_contract import RUN, STATUS, validate_policy, validate_scope_receipt, ARTIFACTS

s = dict(schema='textop.ldm100k_heldout_scope_builder.v1', run=RUN,
         scope_build_authorized=True, generation_authorized=False,
         optimizer_updates_authorized=0, new_gpu_contexts_authorized=0,
         reserve_bytes=2 * 1024**3, metadata_upper_bound_bytes=128 * 1024**2,
         log_allowance_bytes=128 * 1024**2, minimum_ram_free_bytes=16 * 1024**3,
         minimum_disk_free_bytes=3 * 1024**3, timeout_seconds=900,
         executor_sha256='1' * 64, supervisor_sha256='2' * 64,
         corpus_spec_sha256='3' * 64, heldout_plan_sha256='4' * 64,
         bound_sources={'scripts/fixture.py': '5' * 64},
         verified_model=dict(run='user_ldm100k_prefix_document_v1', spec_sha256='6' * 64,
             receipt_sha256='7' * 64, exit_sha256='8' * 64, document_sha256='9' * 64,
             closure={'fixture': True}, document_path='/fixture/SUMMARY.json'))
validate_policy(s)
disabled = copy.deepcopy(s)
disabled.update(scope_build_authorized=False, verified_model=None)
validate_policy(disabled, enabled=False)
base = dict(source_validation_families=1774, seeds=[20261028, 20261029, 20261030, 20261031, 20261101],
    modes=['conditional_cfg5', 'unconditional'], diffusion_steps=5, cfg=5,
    maximum_segment_future_frames=200, segment_count=32, planned_trial_count=320,
    actual_generation_runs=0, actual_model_loads=0, semantic_metrics_computed=False,
    transition_metrics_included=False, source_and_mirror_family_disjoint=True,
    train_families_used_for_eval=False, text_motion_feature_extractor_accepted=False)
r = dict(run=RUN, status=STATUS, output_writers_closed=True, source_sha256=s['executor_sha256'],
    spec_sha256='a' * 64, verified_model_receipt_sha256='7' * 64,
    actual_array_deserializations=0, actual_model_loads=0, actual_gpu_contexts=0,
    actual_generation_runs=0, optimizer_updates=0, semantic_metrics_computed=False,
    array_bytes_revalidated_here=False, training_data_or_model_modified=False,
    generation_inputs_numerically_accepted=False, source_array_hash_recheck_required_before_generation=True,
    private_metadata_only=True, actual_validation_original_families=1774,
    artifacts={n: dict(path='/fixture/' + RUN + '/' + n, bytes=100, sha256='b' * 64) for n in ARTIFACTS},
    diagnostic_summary=dict(base, diagnostic_family_limit=32, selected_validation_families=32),
    validation_summary=dict(base, diagnostic_family_limit=None, selected_validation_families=1774))
validate_scope_receipt(r, s, 'a' * 64)


def rejected(call, name):
    try:
        call()
    except (AssertionError, KeyError, TypeError):
        return
    raise AssertionError('Accepted corrupt isolated metadata: ' + name)


count = 0
for key, value in [('scope_build_authorized', False), ('generation_authorized', True),
                   ('verified_model', None), ('reserve_bytes', 1024**3),
                   ('metadata_upper_bound_bytes', 129 * 1024**2),
                   ('minimum_disk_free_bytes', 2 * 1024**3), ('timeout_seconds', 901),
                   ('executor_sha256', 'f' * 63), ('optimizer_updates_authorized', 1),
                   ('new_gpu_contexts_authorized', 1)]:
    bad = copy.deepcopy(s)
    bad[key] = value
    rejected(lambda: validate_policy(bad), key)
    count += 1
for key, value in [('status', 'WRONG'), ('output_writers_closed', False),
                   ('actual_model_loads', 1), ('actual_array_deserializations', 1),
                   ('actual_gpu_contexts', 1), ('optimizer_updates', 1),
                   ('actual_generation_runs', 1), ('actual_generation_runs', False),
                   ('semantic_metrics_computed', True), ('array_bytes_revalidated_here', True),
                   ('source_array_hash_recheck_required_before_generation', False),
                   ('generation_inputs_numerically_accepted', True),
                   ('verified_model_receipt_sha256', '0' * 64)]:
    bad = copy.deepcopy(r)
    bad[key] = value
    rejected(lambda: validate_scope_receipt(bad, s, 'a' * 64), key)
    count += 1
for label, change in [
    ('missing-private-index', lambda x: x['artifacts'].pop('source_index.private.json')),
    ('payload-budget', lambda x: x['artifacts']['source_index.private.json'].update(bytes=129 * 1024**2)),
    ('cfg-change', lambda x: x['diagnostic_summary'].update(cfg=4)),
    ('invented-transition', lambda x: x['validation_summary'].update(transition_metrics_included=True)),
    ('incorrect-trial-count', lambda x: x['diagnostic_summary'].update(planned_trial_count=319)),
    ('wrong-seeds', lambda x: x['diagnostic_summary'].update(seeds=[1])),
    ('subset-replaces-all-val', lambda x: x['validation_summary'].update(diagnostic_family_limit=32)),
    ('train-leak-claim', lambda x: x['validation_summary'].update(train_families_used_for_eval=True)),
]:
    bad = copy.deepcopy(r)
    change(bad)
    rejected(lambda: validate_scope_receipt(bad, s, 'a' * 64), label)
    count += 1
print(json.dumps(dict(status='LDM100K_SCOPE_NEW_ISOLATED_CONTRACT_FIXTURES_PASSED',
    negative_cases=count, real_scope_builds=0, real_models_loaded=0,
    production_arrays_read=0, gpu_or_sdk_jobs=0, actual_runtime_acceptance=False)))
