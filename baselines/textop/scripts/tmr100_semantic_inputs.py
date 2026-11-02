"""Frozen trained-evaluator and closed generated diagnostic provenance, no models."""
import pathlib
import hashlib
import json
from tmr100_trained_acceptance import (ROOT,PARENT,owned,read,sha,accepted_frozen_export,
    closed_evaluator_job)

PROTOCOL=dict(encoding_sample_mean=True,latent_dimensions=256,per_sample_L2_normalization=False,
    distance='raw_latent_Euclidean_float64',candidate_count=32,
    candidate_schedule='sha256_segment_order_disjoint_blocks_with_fixed_tail_distractors_v1',
    tied_rank='zero_based_average_of_exact_distance_ties',positive='exact_paired_annotation_segment_only',
    all188_queries_scored=True,covariance_ddof=1,fid_population_per_seed=188,
    pooled_repeated_reference_across_seeds=False,seed_aggregation='equal_mean_all_five_seeds_no_selection',
    scored_frames='only_declared_future_no_history_no_padding_tail',
    continuous_transitions_or_full_validation=False,author_exact_protocol_claimed=False)


def validate_semantic_policy(policy):
    assert policy['evaluator_training_epochs']==100 and policy['evaluator_optimizer_updates']==101500
    assert policy['schema']=='textop.trained_robot_tmr100_semantic.v1'
    assert policy['metric_protocol']==PROTOCOL
    assert type(policy['witnesses_authorized']) is bool and type(policy['metrics_authorized']) is bool
    assert not (policy['witnesses_authorized'] and policy['metrics_authorized'])
    if not policy['witnesses_authorized'] and not policy['metrics_authorized']:
        assert policy['frozen_model_binding'] is None and policy['witness_bindings'] is None
    else:
        assert isinstance(policy['frozen_model_binding'],dict) and policy['frozen_model_binding']
        if policy['metrics_authorized']:assert isinstance(policy['witness_bindings'],dict) and policy['witness_bindings']
        else:assert policy['witness_bindings'] is None
    assert policy['reserve_bytes']==2*1024**3 and policy['log_allowance_bytes']==128*1024**2
    assert policy['output_upper_bytes']==8*1024**2
    assert policy['seeds']==[20261028,20261029,20261030,20261031,20261101]
    assert policy['modes']==['conditional_cfg5','unconditional']
    return policy


def accepted_frozen_verifier(binding):
    assert binding['run']=='tmr100_frozen_verify_v1'
    sp=owned(ROOT/'configs/tmr100-frozen-verify-spec.json')
    assert sha(sp)==binding['spec_sha256']
    s=read(sp)
    assert s['authorized'] is True and s['role']=='verify'
    for name,value in s['bound_sources'].items():assert sha(owned(ROOT/name))==value
    assert sha(owned(ROOT/'scripts/run_tmr100_frozen_cpu.py'))==s['executor_sha256']
    assert sha(owned(ROOT/'scripts/supervise_tmr100_frozen_cpu.py'))==s['supervisor_sha256']
    rp=owned(PARENT/binding['run']/'receipt.json')
    assert sha(rp)==binding['receipt_sha256']
    rec=read(rp)
    assert rec['status']=='TMR100_TRAINED_FROZEN_FRESH_CPU_VERIFIED' and rec['output_writers_closed'] is True
    assert rec['source_sha256']==s['executor_sha256'] and rec['spec_sha256']==sha(sp)
    for name in ['full_model_and_cpu_adamw_restored_exact','independent_inference_reload_exact',
                 'synthetic_mean_forward_exact','original_fullstate_unchanged']:
        assert rec[name] is True
    assert rec['training_optimizer_updates_performed']==rec['new_cuda_contexts']==0
    assert rec['cuda_rng_restored'] is False and rec['semantic_metrics_accepted'] is False
    closure=closed_evaluator_job(binding['run'],rp,sha(rp),s['supervisor_sha256'],sha(sp))
    assert binding['closure']==closure and binding['export_binding']==s['export_binding']
    exported=accepted_frozen_export(s['export_binding'])
    for name in ['training_binding','inference','trained_model_sha256','fullstate_metadata_audit','synthetic_probe']:
        assert rec[name]==exported['receipt'][name],name
    return dict(binding=binding,spec=s,receipt=rec,export=exported)


def runtime_signature(policy):
    validate_semantic_policy(policy)
    frozen={k:v for k,v in policy.items() if k not in ['witnesses_authorized','metrics_authorized','witness_bindings']}
    return hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def semantic_closure(run,request_sha,supervisor_sha):
    log=owned(PARENT/(run+'_supervision'))
    paths=dict(receipt=owned(PARENT/run/'receipt.json'),exit=log/'exit.json',process=log/'process_exit.json',
               resources=log/'resources.jsonl',outer=ROOT/'receipts'/('tmr_semantic_'+run+'_outer_exit.json'))
    for path in paths.values():owned(path)
    ex,pr,outer=[read(paths[k]) for k in ['exit','process','outer']]
    assert type(outer['actual_returncode']) is int and outer['actual_returncode']==0
    for value in [ex,pr]:
        assert value['supervision_passed'] is True and value['child_reaped'] is True and value['child_exit_code']==0
        assert value['timed_out'] is False and value['supervisor_error'] is None
        assert value['sample_errors']==[] and value['shutdown_actions']==[] and not value.get('cleanup_error')
    assert ex['source_sha256']==supervisor_sha and ex['request_sha256']==request_sha
    for key,field in [('receipt','receipt_sha256'),('process','process_exit_sha256'),('resources','resource_sha256')]:
        assert sha(paths[key])==ex[field]
    samples=[json.loads(x) for x in paths['resources'].read_text().splitlines()]
    assert len(samples)==ex['samples_written']==pr['samples_written'] and len(samples)>=3
    assert samples[-1]['phase']=='after_exit' and samples[-1]['owned_group_processes']==[]
    return {k+'_sha256':sha(v) for k,v in paths.items()}


def accepted_semantic_witnesses(binding,policy):
    assert set(binding['runs'])=={'tmr100_semantic_author_v1','tmr100_semantic_document_v1'}
    assert binding['runtime_binding_sha256']==runtime_signature(policy)
    records=[]
    for run,item in binding['runs'].items():
        request=owned(ROOT/'receipts'/('tmr_semantic_'+run+'_request.json'))
        assert sha(request)==item['request_sha256']
        rp=owned(PARENT/run/'receipt.json');assert sha(rp)==item['receipt_sha256']
        rec=read(rp)
        assert rec['run']==run and rec['mode']=='witness'
        assert rec['status']=='TMR100_REAL_SEMANTIC_EMBEDDING_WITNESS_FINISHED' and rec['output_writers_closed'] is True
        assert rec['runtime_binding_sha256']==runtime_signature(policy)
        assert rec['source_sha256']==policy['executor_sha256'] and rec['request_sha256']==sha(request)
        assert rec['actual_completed_trials']==2 and rec['actual_segments']==1
        assert rec['source_model_and_arrays_unchanged'] and rec['inference_rng_unchanged']
        assert rec['text_truncation_performed'] is False and rec['stochastic_sampling_or_dropout_used'] is False
        assert rec['formal_semantic_scores_computed'] is False and rec['training_optimizer_updates']==rec['generator_teacher_new_updates']==0
        assert semantic_closure(run,sha(request),policy['supervisor_sha256'])==item['closure']
        for artifact in rec['private_artifacts'].values():
            path=owned(artifact['path']);assert path.parent==rp.parent
            assert path.stat().st_size==artifact['bytes'] and sha(path)==artifact['sha256']
        records.append(rec)
    for name in ['mean_latent_arrays','actual_seeds','actual_modes','maximum_actual_text_tokens','actual_completed_trials']:
        assert records[0][name]==records[1][name],name
    return records


def accepted_diagnostic_metadata(binding):
    from ldm100k_scope_acceptance import accepted_scope
    from filtered15000_vae_completion import closed_job
    gs=owned(ROOT/'configs/ldm100k-generation-spec.json')
    assert sha(gs)==binding['generation_spec_sha256']
    generation_spec=read(gs)
    scope=accepted_scope(generation_spec['scope_binding'])
    origin=owned(ROOT/'ldm100k_user_handoff_20261028/user_ldm100k_diagnostic32_v1')
    rp=owned(origin/'receipt.json');assert sha(rp)==binding['generation_receipt_sha256']
    rec=read(rp)
    assert rec['status']=='USER_LDM100K_DIAGNOSTIC32_GENERATION_FINISHED' and rec['output_writers_closed'] is True
    assert rec['completed_pairs']==rec['planned_pairs']==940 and rec['completed_trials']==rec['planned_trials']==1880
    assert rec['failed_pairs']==rec['unattempted_pairs']==rec['training_optimizer_updates']==0
    assert rec['model_and_source_bytes_unchanged'] is True
    closure=closed_job('ldm100k_generation_'+rec['run'],rp,sha(rp),generation_spec['supervisor_sha256'])
    assert closure==binding['generation_closure']
    journal=owned(origin/'pairs.private.jsonl')
    assert sha(journal)==rec['private_journal_sha256']==binding['private_journal_sha256']
    expected={(x['segment_id'],seed) for x in scope['scope']['segments'] for seed in scope['scope']['summary']['seeds']}
    assert len(expected)==940 and len(scope['scope']['segments'])==188
    started,finished=set(),{}
    with journal.open(encoding='utf-8') as stream:
        for line in stream:
            item=__import__('json').loads(line);key=(item['segment_id'],item['seed'])
            assert key in expected
            if item['status']=='PAIR_STARTED':
                assert key not in started and key not in finished;started.add(key)
            else:
                assert item['status']=='PAIR_FINISHED_SAME_NOISE_VERIFIED' and key in started and key not in finished
                finished[key]=item
    assert started==set(finished)==expected
    return dict(scope=scope['scope'],source_index=scope['source_index'],receipt=rec,
                generation_spec=generation_spec,finished=finished,origin=origin,journal=journal,
                normalization_sha256=scope['receipt']['normalization_sha256_bound'])
