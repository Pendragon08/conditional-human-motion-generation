"""Separate10k paired replay provenance, never impersonating5k or full40k."""
import hashlib
import json
import pathlib
from quick_pilot_protocol import select_sources as baseline_select_sources
from teacher_evaluation_control import validate_backup
from full_teacher_backup_protocol import RUN

QUALITY='TEACHER10000_DIAGNOSTIC_NOT_PAPER_COMPLETE'
STATUS='TEXTOP_TEACHER10000_PILOT_EXECUTION_COMPLETED'

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def gate(stage,spec):
    assert stage in ['smoke','sample'] and spec['schema']=='textop.teacher10000_pilot.v1'
    assert spec['quality_label']==QUALITY and spec['checkpoint_completed']==10000 and spec['checkpoint_updates']==200000
    assert spec['pilot_launch_enabled'] is True,'10k replay disabled before SDK dispatch'
    if stage=='sample':assert spec['sample_launch_enabled'] is True,'Actual author/fresh smoke required'

def authorize_teacher(spec,teacher):
    r,stop,backup,control,resume=[teacher[k] for k in ['training','stop','backup','control','resume_spec']]
    digests=teacher['artifact_sha256']
    assert spec['teacher_run']==control['run']==resume['run']==stop['run']==RUN
    assert control['handoff_enabled'] and control['evaluation_checkpoint']==10000 and control['paper_reference_iterations']==40000
    assert digests['control']==spec['control_spec_sha256']==stop['control_spec_sha256']
    assert digests['resume_spec']==control['resume_spec_sha256']
    assert stop['status']=='USER_AUTHORIZED_TEACHER10000_EVALUATION_HANDOFF'
    assert stop['action']=='user_authorized_pause_for_10000_physics_evaluation'
    assert stop['recoverable_completed']==10000 and stop['recoverable_optimizer_updates']==200000
    assert stop['common_sdk_lock_released'] is True and stop['paper_40000_completed'] is False
    assert stop['runtime_modified'] is False and stop['signal_identity']=='pidfd_original_process'
    assert stop['signals_sent'] and stop['signals_sent'][0]['signal']=='SIGINT'
    assert all(x['pid']==stop['trainer_pid'] and x['signal'] in ['SIGINT','SIGTERM','SIGKILL'] for x in stop['signals_sent'])
    assert stop['final_training_receipt_sha256']==digests['training'] and stop['backup_receipt_sha256']==digests['backup']
    for key in ['process','supervisor','outer']:
        assert stop['original_exits'][key]==teacher[key] and stop['original_exit_sha256'][key]==digests[key]
    assert teacher['process']['child_pid']==stop['owned_group'] and teacher['process']['child_reaped']
    assert not teacher['process']['timed_out'] and teacher['process']['supervisor_error'] is None
    # The original40k outer failure on an explicitly authorized10k pause is retained.
    assert type(teacher['outer']['actual_returncode']) is int
    assert r['mode']=='formal' and r['status'] in ['FAILED','RUNNING_FORMAL_FILTER_TRAINING']
    assert r['resume_exact_state_verified'] and r['num_envs']==8192 and r['reference_sources']==8278
    assert 10000<=r['completed_learning_iterations']<40000 and r['optimizer_updates']==20*r['completed_learning_iterations']
    assert r['reference_fingerprint']==spec['expected_reference_fingerprint']
    meta=r['latest_checkpoint'];assert meta==stop['checkpoint']
    assert meta['name']=='completed_10000.pt' and meta['completed_learning_iterations']==10000 and meta['actual_optimizer_updates']==200000
    assert meta['sha256']==spec['expected_checkpoint_sha256'] and 0<meta['bytes']<=128*1024**2
    validate_backup(control,backup,meta)
    assert backup['audit']['reference_fingerprint']==r['reference_fingerprint']
    for k in ['plan_sha256','source_sha256','runtime_sha256','reference_fingerprint']:
        assert r[k]==teacher['backup_spec']['training_receipt_bindings'][k]
    return backup['audit']['full_checkpoint']

def load_teacher(root,spec):
    root=pathlib.Path(root);prefix='full_resume_'+RUN
    assert sha(root/'scripts/quick_pilot_protocol.py')==spec['selection_source_sha256']
    paths={'training':root/'tracker_validation'/RUN/'receipt.json',
      'stop':root/'receipts/teacher10000_controlled_stop.json',
      'backup':root/'receipts/teacher10000_backup_verified.json',
      'control':root/'configs/teacher10000-evaluation-spec.json',
      'resume_spec':root/'configs/full-teacher-resume-spec.json',
      'backup_spec':root/'configs/full-teacher-backup-spec.json'}
    paths.update({k:root/'receipts'/(prefix+suffix) for k,suffix in [('process','_process_exit.json'),('supervisor','_exit.json'),('outer','_outer_exit.json')]})
    t={k:json.loads(p.read_bytes()) for k,p in paths.items()};t['artifact_sha256']={k:sha(p) for k,p in paths.items()}
    assert t['artifact_sha256']['backup_spec']==t['control']['backup_spec_sha256']
    for n,h in t['resume_spec']['frozen_bindings'].items():assert sha(root/n)==h,n
    assert sha(root/'scripts/supervise_full_teacher_resume.py')==t['resume_spec']['source_sha256']
    for n,h in t['control']['bound_control_sources'].items():assert sha(root/n)==h,n
    authorize_teacher(spec,t)
    return t

def select_sources(rows,stage,spec):
    selected=baseline_select_sources(rows,stage,spec)
    root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
    p=root/'tracker_validation/quick_pilot_selection_metadata_v1/selection.private.json'
    assert sha(p)==spec['selection_sha256'];original=json.loads(p.read_text())['stages'][stage]
    assert selected==[x['global_index'] for x in original]
    for i,x in zip(selected,original):
        assert rows[i]['canonical_source']==x['canonical_source'] and rows[i]['sha256']==x['reference_sha256'] and rows[i]['frames']==x['frames'] and rows[i]['split']==x['split']
    return selected

def settings_signature(spec):
    keys=['device','torch_memory_fraction','cpu_threads','seed','control_hz','smoke_parallel_envs','smoke_max_control_calls','smoke_timeout_seconds','sample_parallel_envs','sample_source_count','sample_max_control_calls','sample_timeout_seconds','expected_reference_sources','expected_reference_fingerprint','strict_pelvis_height_threshold_m','strict_end_effector_height_threshold_m','checkpoint_completed','checkpoint_updates','selection_sha256']
    return hashlib.sha256(json.dumps({k:spec[k] for k in keys},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_smoke_witnesses(root, spec, checkpoint_sha):
    from strict_tracking_coverage import verify_trial
    root = pathlib.Path(root)
    witnesses = spec['smoke_witnesses']
    assert len(witnesses) == 2 and len({w['run'] for w in witnesses}) == 2
    for w in witnesses:
        out = root/'tracker_validation'/w['run']
        rp = out/'receipt.json';ep = root/'receipts'/('teacher10000_pilot_'+w['run']+'_exit.json')
        sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256']
        r=json.loads(rp.read_text());e=json.loads(ep.read_text())
        assert r['status']==STATUS and r['mode']=='teacher10000_pilot' and r['stage']=='smoke'
        assert r['quality_label']==QUALITY and r['output_writers_closed'] is True
        assert r['optimizer_updates']==0 and r['dataset_accepted'] is False and r['physical_filter_executed'] is False
        assert r['policy_checkpoint_sha256']==checkpoint_sha and r['policy_checkpoint_completed_iterations']==10000
        assert r['policy_checkpoint_optimizer_updates']==200000
        assert r['source_sha256']==spec['executor_sha256'] and r['execution_settings_sha256']==settings_signature(spec)
        assert r['evaluation_protocol_sha256']==spec['protocol_sha256']
        assert r['trace_source_sha256']==spec['trace_sha256']
        assert r['terminal_execution_state_captured_before_autoreset'] is True
        assert r['replayed_original_sources']==r['finished_trials']==8 and r['complete_trials']>=1
        assert 0<r['actual_vectorized_control_calls']<=1024 and r['actual_scene_physics_steps']==4*r['actual_vectorized_control_calls']
        assert e['supervision_passed'] is True and e['child_exit_code']==0 and e['timed_out'] is False
        assert e['evaluation_receipt_sha256']==sha(rp) and e['numerical_status']==STATUS
        process=root/'receipts'/('teacher10000_pilot_'+w['run']+'_process_exit.json')
        samples=root/'logs'/('teacher10000_pilot_'+w['run']+'_resources.jsonl')
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
