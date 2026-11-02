"""Pure signed policy/output checks for the private CPU metadata scope job."""
RUN = 'user_ldm100k_heldout_scope_v1'
STATUS = 'USER_LDM100K_HELDOUT_METADATA_SCOPE_BUILT'
ARTIFACTS = {'diagnostic_scope.private.json', 'validation_scope.private.json',
             'source_index.private.json'}
SEEDS = [20261028, 20261029, 20261030, 20261031, 20261101]


def digest_string(value):
    assert isinstance(value, str) and len(value) == 64
    assert all(c in '0123456789abcdef' for c in value)


def validate_policy(spec, enabled=True):
    assert spec['schema'] == 'textop.ldm100k_heldout_scope_builder.v1'
    assert spec['run'] == RUN and spec['scope_build_authorized'] is enabled
    assert spec['generation_authorized'] is False
    assert spec['optimizer_updates_authorized'] == spec['new_gpu_contexts_authorized'] == 0
    assert spec['reserve_bytes'] == 2 * 1024**3
    assert type(spec['metadata_upper_bound_bytes']) is int
    assert 0 < spec['metadata_upper_bound_bytes'] <= 128 * 1024**2
    assert spec['log_allowance_bytes'] == 128 * 1024**2
    assert spec['minimum_ram_free_bytes'] == 16 * 1024**3
    assert spec['minimum_disk_free_bytes'] >= (spec['reserve_bytes'] +
           spec['metadata_upper_bound_bytes'] + spec['log_allowance_bytes'])
    assert spec['timeout_seconds'] == 900
    for key in ['executor_sha256', 'supervisor_sha256', 'corpus_spec_sha256',
                'heldout_plan_sha256']:
        digest_string(spec[key])
    assert spec['bound_sources']
    for key, value in spec['bound_sources'].items():
        assert key.startswith(('scripts/', 'configs/', 'upstream/'))
        assert '..' not in key.split('/') and not key.startswith('/')
        digest_string(value)
    if enabled:
        binding = spec['verified_model']
        assert binding['run'] == 'user_ldm100k_prefix_document_v1'
        for key in ['spec_sha256', 'receipt_sha256', 'exit_sha256', 'document_sha256']:
            digest_string(binding[key])
        assert binding['closure'] and binding['document_path'].startswith('/')
    else:
        assert spec['verified_model'] is None


def validate_scope_receipt(receipt, spec, expected_spec_sha):
    validate_policy(spec)
    r = receipt
    assert r['run'] == RUN and r['status'] == STATUS and r['output_writers_closed'] is True
    assert r['source_sha256'] == spec['executor_sha256'] and r['spec_sha256'] == expected_spec_sha
    assert r['verified_model_receipt_sha256'] == spec['verified_model']['receipt_sha256']
    for key in ['actual_array_deserializations', 'actual_model_loads', 'actual_gpu_contexts',
                'actual_generation_runs', 'optimizer_updates']:
        assert type(r[key]) is int and r[key] == 0
    for key in ['semantic_metrics_computed', 'array_bytes_revalidated_here',
                'training_data_or_model_modified', 'generation_inputs_numerically_accepted']:
        assert r[key] is False
    assert r['source_array_hash_recheck_required_before_generation'] is True
    assert r['private_metadata_only'] is True
    assert set(r['artifacts']) == ARTIFACTS
    total = 0
    for name, artifact in r['artifacts'].items():
        assert artifact['path'].endswith('/' + RUN + '/' + name)
        assert type(artifact['bytes']) is int and artifact['bytes'] > 0
        total += artifact['bytes']
        digest_string(artifact['sha256'])
    assert total <= spec['metadata_upper_bound_bytes']
    for key, limit in [('diagnostic_summary', 32), ('validation_summary', None)]:
        q = r[key]
        assert q['diagnostic_family_limit'] == limit
        assert q['source_validation_families'] == r['actual_validation_original_families']
        assert q['selected_validation_families'] == (limit or r['actual_validation_original_families'])
        assert q['seeds'] == SEEDS and q['modes'] == ['conditional_cfg5', 'unconditional']
        assert q['diffusion_steps'] == q['cfg'] == 5
        assert q['maximum_segment_future_frames'] == 200
        assert q['planned_trial_count'] == q['segment_count'] * 10
        assert q['actual_generation_runs'] == q['actual_model_loads'] == 0
        assert q['semantic_metrics_computed'] is False and q['transition_metrics_included'] is False
        assert q['source_and_mirror_family_disjoint'] is True and q['train_families_used_for_eval'] is False
        assert q['text_motion_feature_extractor_accepted'] is False
