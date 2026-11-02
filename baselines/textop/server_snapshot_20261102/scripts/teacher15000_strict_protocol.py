"""Explicit user-directed all-source strict replay at15k, separate from EMA gate."""
import hashlib,json,pathlib
from filter_recovery_backup_protocol import RUN
from teacher15000_evaluation_control import validate_backup
QUALITY='USER_DIRECT_STRICT_15000_NOT_PAPER_TWO_STAGE_FILTER'
STATUS='TEXTOP_TEACHER15000_DIRECT_STRICT_EXECUTION_COMPLETED'
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def gate(stage,spec):
    assert stage in ['smoke','full'] and spec['schema']=='textop.teacher15000_direct_strict.v1'
    assert spec['quality_label']==QUALITY and spec['checkpoint_completed']==15000 and spec['checkpoint_updates']==300000
    assert spec['pilot_launch_enabled'] is True
    if stage=='full':assert spec['full_launch_enabled'] is True
def authorize_teacher(spec,t):
    r,stop,b,c,resume=[t[k] for k in ['training','stop','backup','control','resume_spec']]
    h=t['artifact_sha256']
    assert spec['teacher_run']==c['run']==resume['run']==stop['run']==RUN
    assert h['control']==spec['control_spec_sha256']==stop['control_spec_sha256']
    assert h['resume_spec']==c['resume_spec_sha256']
    assert stop['status']=='USER_AUTHORIZED_TEACHER15000_EVALUATION_HANDOFF'
    assert stop['recoverable_completed']==15000 and stop['recoverable_optimizer_updates']==300000
    assert stop['common_sdk_lock_released'] and not stop['paper_40000_completed'] and not stop['runtime_modified']
    assert stop['signal_identity']=='pidfd_original_process' and stop['signals_sent'][0]['signal']=='SIGINT'
    assert stop['final_training_receipt_sha256']==h['training'] and stop['backup_receipt_sha256']==h['backup']
    for k in ['process','supervisor','outer']:
        assert stop['original_exits'][k]==t[k] and stop['original_exit_sha256'][k]==h[k]
    assert t['process']['child_pid']==stop['owned_group'] and t['process']['child_reaped']
    assert not t['process']['timed_out'] and t['process']['supervisor_error'] is None
    assert type(t['outer']['actual_returncode']) is int
    assert r['mode']=='formal' and r['resume_exact_state_verified'] and r['num_envs']==8192 and r['reference_sources']==8278
    assert 15000<=r['completed_learning_iterations']<40000 and r['optimizer_updates']==20*r['completed_learning_iterations']
    assert r['reference_fingerprint']==spec['expected_reference_fingerprint']
    assert r['latest_checkpoint']==stop['checkpoint'] and r['latest_checkpoint']['name']=='completed_15000.pt'
    assert r['latest_checkpoint']['sha256']==spec['expected_checkpoint_sha256']
    validate_backup(c,b,stop['checkpoint'])
    return b['audit']['full_checkpoint']
def load_teacher(root,spec):
    root=pathlib.Path(root);prefix='filter_recovery_'+RUN
    paths=dict(training=root/'tracker_validation'/RUN/'receipt.json',stop=root/'receipts/teacher15000_controlled_stop.json',
        backup=root/'receipts/teacher15000_backup_verified.json',control=root/'configs/teacher15000-direct-evaluation-spec.json',
        resume_spec=root/'configs/teacher10000-filter-recovery-spec.json',backup_spec=root/'configs/filter-recovery-backup-spec.json')
    paths.update({k:root/'receipts'/(prefix+s) for k,s in [('process','_process_exit.json'),('supervisor','_exit.json'),('outer','_outer_exit.json')]})
    t={k:json.loads(p.read_bytes()) for k,p in paths.items()};t['artifact_sha256']={k:sha(p) for k,p in paths.items()}
    assert t['artifact_sha256']['backup_spec']==t['control']['backup_spec_sha256']
    for n,h in t['resume_spec']['frozen_bindings'].items():assert sha(root/n)==h,n
    assert sha(root/'scripts/supervise_teacher10000_filter_recovery.py')==t['resume_spec']['source_sha256']
    for n,h in t['control']['bound_control_sources'].items():assert sha(root/n)==h,n
    authorize_teacher(spec,t);return t
def select_sources(rows,stage,spec):
    assert len(rows)==len({r['canonical_source'] for r in rows})==8278
    if stage=='full':
        # Outcome-independent duration ordering reduces padding; every source appears once.
        return sorted(range(8278),key=lambda i:(rows[i]['frames'],i))
    root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
    p=root/'tracker_validation/quick_pilot_selection_metadata_v1/selection.private.json'
    assert sha(p)==spec['selection_sha256']
    original=json.loads(p.read_text())['stages']['smoke'];selected=[x['global_index'] for x in original]
    assert len(selected)==len(set(selected))==8
    for i,x in zip(selected,original):
        assert rows[i]['canonical_source']==x['canonical_source'] and rows[i]['sha256']==x['reference_sha256'] and rows[i]['frames']==x['frames'] and rows[i]['split']==x['split']
    return selected
def settings_signature(s):
    keys=['device','torch_memory_fraction','cpu_threads','seed','control_hz','smoke_parallel_envs','smoke_max_control_calls','smoke_timeout_seconds',
        'full_parallel_envs','full_source_count','full_max_control_calls','full_timeout_seconds','expected_reference_sources','expected_reference_fingerprint',
        'strict_pelvis_height_threshold_m','strict_end_effector_height_threshold_m','checkpoint_completed','checkpoint_updates','selection_sha256',
        'maximum_private_trace_bytes','trace_sha256','full_source_order','strict_rollouts_per_motion']
    return hashlib.sha256(json.dumps({k:s[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_smoke_witnesses(root,spec,checkpoint_sha):
    from strict_tracking_coverage import verify_trial
    root=pathlib.Path(root);witnesses=spec['smoke_witnesses']
    assert len(witnesses)==len({w['run'] for w in witnesses})==2
    for w in witnesses:
        out=root/'tracker_validation'/w['run'];rp=out/'receipt.json';ep=root/'receipts'/('teacher15000_strict_'+w['run']+'_exit.json')
        assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256']
        r=json.loads(rp.read_text());e=json.loads(ep.read_text())
        assert r['status']==STATUS and r['mode']=='teacher15000_strict' and r['stage']=='smoke' and r['output_writers_closed']
        assert r['quality_label']==QUALITY and r['optimizer_updates']==0 and not r['dataset_accepted']
        assert r['policy_checkpoint_sha256']==checkpoint_sha and r['policy_checkpoint_completed_iterations']==15000 and r['policy_checkpoint_optimizer_updates']==300000
        assert r['source_sha256']==spec['executor_sha256'] and r['evaluation_protocol_sha256']==spec['protocol_sha256'] and r['trace_source_sha256']==spec['trace_sha256']
        assert r['execution_settings_sha256']==settings_signature(spec)
        assert r['finished_trials']==r['replayed_original_sources']==8 and r['complete_trials']>=1
        assert r['actual_scene_physics_steps']==4*r['actual_vectorized_control_calls'] and r['terminal_execution_state_captured_before_autoreset']
        assert e['child_exit_code']==0 and e['child_reaped'] and e['supervision_passed'] and not e['timed_out'] and e['supervisor_error'] is None
        assert e['evaluation_receipt_sha256']==sha(rp)
        prefix='teacher15000_strict_'+w['run']
        outer=json.loads((root/'receipts'/(prefix+'_outer_exit.json')).read_text());assert outer['actual_returncode']==0
        process=root/'receipts'/(prefix+'_process_exit.json');samples=root/'logs'/(prefix+'_resources.jsonl')
        assert sha(process)==e['process_exit_sha256'] and sha(samples)==e['resources_sha256']
        pr=json.loads(process.read_text());assert pr['child_exit_code']==0 and pr['child_reaped'] and not pr['timed_out']
        assert not pr.get('cleanup_error') and pr['sample_errors']==[]
        sr=[json.loads(x) for x in samples.read_text().splitlines()];assert len(sr)==pr['samples_written'] and sr[-1]['phase']=='after_exit'
        strict=out/'strict.private.json';journal=out/'strict_trials.private.jsonl';index=out/'executed_traces.private.json'
        assert sha(strict)==r['private_strict_log_sha256'] and sha(journal)==r['private_trial_journal_sha256'] and sha(index)==r['private_executed_trace_index_sha256']
        rows=json.loads(strict.read_text())['motions'];records=json.loads(index.read_text());assert len(rows)==len(records)==8
        expected=[]
        for row,record in zip(rows,records):
            # Finishing order may differ, so join by private motion identity below.
            assert len(row['trials'])==1;trial=row['trials'][0];verify_trial(trial,trial['reference_frames'],0)
            expected.append(dict(motion_id=row['motion_id'],trial=trial))
        enc=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
        assert sorted(map(enc,expected))==sorted(map(enc,map(json.loads,journal.read_text().splitlines())))
        ids=[]
        for record in records:
            folder=pathlib.Path(record['folder']);assert folder.resolve().parent==out/'execution_private'
            m=json.loads((folder/'trace.private.json').read_text());binary=folder/'qpos.private.bin'
            assert m['status']=='FINISHED_PHYSICAL_TRIAL' and m['runtime_kind']=='isaaclab_physics'
            assert m['samples']==record['samples']==m['trial']['control_steps']+1
            assert binary.stat().st_size==m['binary_bytes']==m['samples']*36*4
            assert sha(binary)==m['binary_sha256']==record['binary_sha256']
            ids.append(m['motion_id'])
        assert sorted(ids)==sorted(x['motion_id'] for x in rows)
