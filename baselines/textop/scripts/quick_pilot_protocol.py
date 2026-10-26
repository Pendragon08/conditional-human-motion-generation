"""Separate5k pilot provenance/selection; never authorizes full-paper acceptance."""
import hashlib
import json
import pathlib
from quick_budget_guards import OUT, RUN, PLAN_SHA, validate_backup

QUALITY = 'QUICK_REPRODUCTION_NOT_PAPER_COMPLETE'
STATUS = 'TEXTOP_QUICK_PILOT_EXECUTION_COMPLETED'


def gate(stage, spec):
    assert stage in ['smoke', 'sample']
    assert spec['quality_label'] == QUALITY and spec['schema'] == 'textop.quick_pilot.v1'
    assert spec['pilot_launch_enabled'] is True, 'Quick pilot disabled; no SDK dispatched'
    if stage == 'sample':
        assert spec['sample_launch_enabled'] is True, 'Representative pilot still requires actual smoke witnesses'


def authorize_teacher(spec, teacher):
    r, stop, backup, qp, ex = [teacher[k] for k in ['training', 'stop', 'backup', 'quick_plan', 'original_exit']]
    digests = teacher['artifact_sha256']
    assert qp['requested_run'] == RUN and qp['quality_label'] == QUALITY
    assert qp['quick_target_iterations'] == 5000 and qp['paper_target_iterations'] == 40000
    assert qp['early_stop_enabled'] is True
    assert spec['quick_plan_sha256'] == digests['quick_plan'] == stop['plan_sha256']
    assert stop['status'] == 'USER_AUTHORIZED_QUICK_5000_EARLY_STOP_COMPLETE'
    assert stop['action'] == 'user_authorized_early_stop_after_backup'
    assert stop['quality_label'] == QUALITY and stop['user_request'] == qp['user_request']
    assert stop['full_paper_training_completed'] is False and stop['checkpoint_retained'] is True
    assert stop['quick_checkpoint_completed_iterations'] == 5000
    assert stop['signal_identity'] == 'pidfd_original_process' and stop['trainer_start_ticks'] > 0
    signals = stop['signals_sent']
    assert signals and signals[0]['signal'] == 'SIGINT'
    assert all(x['pid'] == stop['trainer_pid'] and x['signal'] in ['SIGINT', 'SIGTERM', 'SIGKILL'] for x in signals)
    assert stop['original_supervisor_exit'] == ex and ex['child_pid'] == stop['owned_group']
    assert ex['timed_out'] is False and ex['numerical_status'] == r['status']
    assert stop['original_final_receipt_sha256'] == digests['training']
    assert stop['original_exit_receipt_sha256'] == digests['original_exit']
    assert stop['backup_receipt_sha256'] == digests['backup']
    assert r['mode'] == 'formal' and r['status'] in ['FAILED', 'RUNNING_FORMAL_FILTER_TRAINING']
    assert r['plan_sha256'] == PLAN_SHA and r['num_envs'] == 8192
    assert 5000 <= r['completed_learning_iterations'] < 40000 and 100000 <= r['optimizer_updates'] < 800000
    assert r['reference_sources'] == spec['expected_reference_sources'] == 8278
    assert r['reference_fingerprint'] == spec['expected_reference_fingerprint']
    metadata = r['latest_checkpoint']
    assert metadata['name'] == 'completed_5000.pt' and metadata['completed_learning_iterations'] == 5000
    assert metadata['actual_optimizer_updates'] == 100000
    checkpoint = stop['checkpoint']
    assert checkpoint['path'] == OUT + '/completed_5000.pt'
    assert checkpoint['sha256'] == metadata['sha256'] and checkpoint['bytes'] == metadata['bytes']
    validate_backup(backup, checkpoint)
    audit = backup['audit']
    assert audit['all_finite_tensor_count'] > 0
    assert audit['optimizer_step_min'] == audit['optimizer_step_max'] == 100000
    assert audit['plan_sha256'] == PLAN_SHA and audit['reference_fingerprint'] == r['reference_fingerprint']
    return checkpoint


def load_teacher(root, spec):
    root = pathlib.Path(root)
    paths = {'training': pathlib.Path(OUT)/'receipt.json',
             'stop': root/'receipts/quick5000_controlled_stop.json',
             'backup': root/'receipts/quick5000_backup_verified.json',
             'quick_plan': root/'configs/quick-reproduction-plan-20261026.json',
             'original_exit': root/'receipts'/('filter_policy_'+RUN+'_exit.json')}
    teacher = {}
    hashes = {}
    for key, path in paths.items():
        raw = path.read_bytes()
        teacher[key] = json.loads(raw)
        hashes[key] = hashlib.sha256(raw).hexdigest()
    teacher['artifact_sha256'] = hashes
    authorize_teacher(spec, teacher)
    return teacher


def select_sources(rows, stage, spec):
    if stage == 'smoke':
        candidates = [i for i,r in enumerate(rows) if 50 <= r['frames'] <= 500]
        selected = sorted(candidates,key=lambda i:(rows[i]['frames'],i))[:8]
        assert len(selected) == 8
        return selected
    assert stage == 'sample' and spec['sample_source_count'] == 64
    buckets = {}
    for i,row in enumerate(rows):
        frames = row['frames']
        if not 50 <= frames <= 2000:
            continue
        parts = pathlib.PurePosixPath(row['canonical_source'].replace('\\','/')).parts
        group = next(x for x in parts if x not in ['.', '..', '/'])
        duration = 0 if frames <= 250 else 1 if frames <= 750 else 2
        buckets.setdefault((group,duration),[]).append(i)
    for values in buckets.values():
        values.sort(key=lambda i:hashlib.sha256(rows[i]['canonical_source'].encode()).hexdigest())
    selected = []
    while len(selected) < 64:
        changed = False
        for key in sorted(buckets):
            if buckets[key] and len(selected) < 64:
                selected.append(buckets[key].pop(0));changed = True
        assert changed, 'Not enough complete sources in the declared pilot scope'
    assert len(set(selected)) == 64
    return selected


def settings_signature(spec):
    keys = ['device','torch_memory_fraction','cpu_threads','seed','control_hz',
            'smoke_parallel_envs','smoke_max_control_calls','smoke_timeout_seconds',
            'sample_parallel_envs','sample_source_count','sample_max_control_calls','sample_timeout_seconds',
            'expected_reference_sources','expected_reference_fingerprint','strict_pelvis_height_threshold_m',
            'strict_end_effector_height_threshold_m','quick_plan_sha256']
    return hashlib.sha256(json.dumps({k:spec[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_smoke_witnesses(root, spec, checkpoint_sha):
    from strict_tracking_coverage import verify_trial
    root = pathlib.Path(root)
    witnesses = spec['smoke_witnesses']
    assert len(witnesses) == 2 and len({w['run'] for w in witnesses}) == 2
    for w in witnesses:
        out = root/'tracker_validation'/w['run']
        rp = out/'receipt.json';ep = root/'receipts'/('quick_pilot_'+w['run']+'_exit.json')
        sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256']
        r=json.loads(rp.read_text());e=json.loads(ep.read_text())
        assert r['status']==STATUS and r['mode']=='quick_pilot' and r['stage']=='smoke'
        assert r['quality_label']==QUALITY and r['output_writers_closed'] is True
        assert r['optimizer_updates']==0 and r['dataset_accepted'] is False and r['physical_filter_executed'] is False
        assert r['policy_checkpoint_sha256']==checkpoint_sha and r['policy_checkpoint_completed_iterations']==5000
        assert r['source_sha256']==spec['executor_sha256'] and r['execution_settings_sha256']==settings_signature(spec)
        assert r['quick_protocol_sha256']==spec['protocol_sha256']
        assert r['trace_source_sha256']==spec['trace_sha256']
        assert r['terminal_execution_state_captured_before_autoreset'] is True
        assert r['replayed_original_sources']==r['finished_trials']==8 and r['complete_trials']>=1
        assert 0<r['actual_vectorized_control_calls']<=1024 and r['actual_scene_physics_steps']==4*r['actual_vectorized_control_calls']
        assert e['supervision_passed'] is True and e['child_exit_code']==0 and e['timed_out'] is False
        assert e['evaluation_receipt_sha256']==sha(rp) and e['numerical_status']==STATUS
        process=root/'receipts'/('quick_pilot_'+w['run']+'_process_exit.json')
        samples=root/'logs'/('quick_pilot_'+w['run']+'_resources.jsonl')
        assert sha(process)==e['process_exit_sha256'] and sha(samples)==e['resources_sha256']
        pr=json.loads(process.read_text())
        for key in ['child_pid','child_exit_code','child_reaped','timed_out','supervision_passed','supervisor_error','samples_written']:
            assert pr[key]==e[key]
        assert pr['child_reaped'] is True and pr['supervisor_error'] is None
        assert not pr.get('cleanup_error') and pr['sample_errors']==[]
        sr=[json.loads(line) for line in samples.read_text().splitlines()]
        assert len(sr)==pr['samples_written'] and len(sr)>=3
        assert sr[0]['phase']=='before_launch' and sr[1]['phase']=='after_launch' and sr[-1]['phase']=='after_exit'
        strict=out/'strict.private.json';journal=out/'strict_trials.private.jsonl'
        assert sha(strict)==r['private_strict_log_sha256'] and sha(journal)==r['private_trial_journal_sha256']
        log=json.loads(strict.read_text());assert log['diagnostic_only'] is True and len(log['motions'])==8
        expected=[]
        for row in log['motions']:
            assert len(row['trials'])==1
            trial=row['trials'][0];verify_trial(trial,trial['reference_frames'],0)
            expected.append({'motion_id':row['motion_id'],'trial':trial})
        encode=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
        actual=[json.loads(line) for line in journal.read_text().splitlines()]
        assert sorted(map(encode,expected))==sorted(map(encode,actual))
        index=out/'executed_traces.private.json'
        assert sha(index)==r['private_executed_trace_index_sha256']
        records=json.loads(index.read_text());assert len(records)==8
        ids=[]
        for record in records:
            folder=pathlib.Path(record['folder'])
            assert folder.resolve().parent==out/'execution_private'
            meta=json.loads((folder/'trace.private.json').read_text())
            assert meta['status']=='FINISHED_PHYSICAL_TRIAL' and meta['runtime_kind']=='isaaclab_physics'
            assert meta['phase']=='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE'
            assert meta['samples']==meta['trial']['control_steps']+1==record['samples']
            assert sha(folder/'poses.private.bin')==meta['binary_sha256']
            assert sha(folder/'executed_poses.private.npz')==meta['npz_sha256']==record['npz_sha256']
            ids.append(meta['motion_id'])
        assert sorted(ids)==sorted(row['motion_id'] for row in log['motions'])
        assert not (out/'evidence_inputs.private.json').exists(), 'Quick smoke must never emit formal acceptance'
