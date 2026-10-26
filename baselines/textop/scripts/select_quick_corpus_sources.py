"""Identify a scoped nominal-tracking subset, never a training-ready corpus.

Uses closed64-source pilot evidence, not the formal40k acceptor. Original files
and failed/unattempted motions are retained. No model, Torch or SDK is imported.
"""
import argparse, copy, hashlib, json, os, pathlib, re
from strict_tracking_coverage import verify_trial
from quick_pilot_protocol import QUALITY, STATUS, gate, load_teacher, authorize_teacher, select_sources, settings_signature, verify_smoke_witnesses
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def source_id(row):return hashlib.sha256(row['canonical_source'].encode()).hexdigest()
def decisions(rows,indices,motions):
    assert len(set(indices))==len(indices) and all(type(i) is int and 0<=i<len(rows) for i in indices)
    expected={source_id(rows[i]):i for i in indices};assert len(expected)==len(indices)
    assert len(motions)==len(indices) and len({m['motion_id'] for m in motions})==len(motions)
    assert set(expected)=={m['motion_id'] for m in motions},'Selected source identity coverage mismatch'
    result=[]
    for motion in motions:
        row=rows[expected[motion['motion_id']]]
        assert row['split'] in ['train','val'] and row['mirrored'] is False and row['frames']>=35
        assert len(motion['trials'])==1
        trial=motion['trials'][0]
        # Reference length comes from the original manifest, never the trial itself.
        failed=verify_trial(trial,row['frames'],0)
        result.append(dict(reference=copy.deepcopy(row),trial=copy.deepcopy(trial),
                           nominal_tracking_completed=not failed,include_in_candidate_view=not failed,
                           failed_in_this_teacher_trial=failed,absolute_physical_infeasibility_claimed=False,
                           final_reference_endpoint_verified=False,mirrored_variant_assessed=False,
                           dataset_accepted=False,training_ready=False))
    result.sort(key=lambda x:expected[source_id(x['reference'])])
    return result

def verify_exit(root,run,receipt):
    folder=root/'tracker_validation'/run;rp=folder/'receipt.json';prefix='quick_pilot_'+run
    ep=root/'receipts'/(prefix+'_exit.json');pp=root/'receipts'/(prefix+'_process_exit.json');sp=root/'logs'/(prefix+'_resources.jsonl')
    e=json.loads(ep.read_text());p=json.loads(pp.read_text());samples=[json.loads(s) for s in sp.read_text().splitlines()]
    assert e['evaluation_receipt_sha256']==sha(rp) and e['numerical_status']==receipt['status']
    assert e['resources_sha256']==sha(sp) and e['process_exit_sha256']==sha(pp)
    for key in ['command','child_pid','child_exit_code','child_reaped','timed_out','supervision_passed','supervisor_error','samples_written']:
        assert e[key]==p[key]
    assert e['supervision_passed'] is True and e['child_reaped'] is True and e['child_exit_code']==0 and e['timed_out'] is False
    assert e['supervisor_error'] is None and not e.get('cleanup_error') and e['sample_errors']==p['sample_errors']==[]
    assert len(samples)==p['samples_written'] and len(samples)>=3
    assert samples[0]['phase']=='before_launch' and samples[1]['phase']=='after_launch' and samples[-1]['phase']=='after_exit'
    command=p['command'];assert str(root/'scripts/evaluate_quick_pilot.py') in command
    assert command[command.index('--stage')+1]=='sample' and command[command.index('--output')+1]==str(folder)
    return {name:sha(path) for name,path in [('pilot_receipt',rp),('pilot_exit',ep),('process_exit',pp),('resources',sp)]}

def select_actual(root,run,spec):
    pilot=json.loads((root/'configs/quick-pilot-spec.json').read_text());gate('sample',pilot)
    teacher=load_teacher(root,pilot);checkpoint=authorize_teacher(pilot,teacher)
    verify_smoke_witnesses(root,pilot,checkpoint['sha256'])
    for name,value in spec['bound_pilot_runtime_sha256'].items():assert sha(root/name)==value
    folder=root/'tracker_validation'/run;r=json.loads((folder/'receipt.json').read_text())
    assert r['status']==STATUS and r['mode']=='quick_pilot' and r['stage']=='sample' and r['quality_label']==QUALITY
    assert r['output_writers_closed'] is True and r['optimizer_updates']==0
    assert r['physical_tracking_executed'] is True
    assert r['dataset_accepted'] is False and r['physical_filter_executed'] is False and r['whole_library_filtered'] is False
    assert r['policy_checkpoint_sha256']==checkpoint['sha256'] and r['policy_checkpoint_completed_iterations']==5000
    assert r['policy_checkpoint_optimizer_updates']==100000 and r['teacher_artifact_sha256']==teacher['artifact_sha256']
    assert r['source_sha256']==pilot['executor_sha256'] and r['quick_protocol_sha256']==pilot['protocol_sha256']
    assert r['trace_source_sha256']==pilot['trace_sha256'] and r['execution_settings_sha256']==settings_signature(pilot)
    assert r['source_sha256']==sha(root/'scripts/evaluate_quick_pilot.py') and r['quick_protocol_sha256']==sha(root/'scripts/quick_pilot_protocol.py')
    assert r['trace_source_sha256']==sha(root/'scripts/quick_pilot_trace.py')
    assert r['num_envs']==16 and r['finished_trials']==r['replayed_original_sources']==64
    assert r['terminal_execution_state_captured_before_autoreset'] is True and r['final_reference_endpoint_comparison_performed'] is False
    assert 0<r['actual_vectorized_control_calls']<=8192 and r['actual_scene_physics_steps']==4*r['actual_vectorized_control_calls']
    profile=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert r['contact_profile_sha256']==profile(r['contact_profile'])==profile(teacher['training']['contact_profile'])
    bindings=verify_exit(root,run,r)
    plan=json.loads((root/'configs/filter-training-plan.json').read_text());rows=[]
    for name in plan['input_reference_exports']:
        f=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name
        export=json.loads((f/'receipt.json').read_text());manifest=f/'manifest.private.jsonl'
        assert export['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED' and sha(manifest)==export['private_manifest_sha256']
        rows.extend(x for x in map(json.loads,manifest.read_text().splitlines()) if not x['mirrored'])
    fp=hashlib.sha256(json.dumps([dict(sha256=x['sha256'],frames=x['frames']) for x in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert len(rows)==8278 and len({x['canonical_source'] for x in rows})==8278
    assert fp==pilot['expected_reference_fingerprint']==r['reference_fingerprint']
    strict=folder/'strict.private.json';journal=folder/'strict_trials.private.jsonl';index=folder/'executed_traces.private.json'
    assert sha(strict)==r['private_strict_log_sha256'] and sha(journal)==r['private_trial_journal_sha256']
    assert sha(index)==r['private_executed_trace_index_sha256']
    log=json.loads(strict.read_text());assert log['diagnostic_only'] is True and log['quality_label']==QUALITY
    assert log['policy_checkpoint_sha256']==checkpoint['sha256'] and log['reference_fingerprint']==fp and log['contact_profile_sha256']==r['contact_profile_sha256']
    selected=decisions(rows,select_sources(rows,'sample',pilot),log['motions'])
    canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
    expected=[dict(motion_id=source_id(x['reference']),trial=x['trial']) for x in selected]
    assert sorted(map(canonical,expected))==sorted(canonical(json.loads(line)) for line in journal.read_text().splitlines())
    traces=json.loads(index.read_text());assert len(traces)==64
    byid={}
    for entry in traces:
        f=pathlib.Path(entry['folder']);assert f.resolve()==f and f.parent==folder/'execution_private'
        meta=json.loads((f/'trace.private.json').read_text())
        assert meta['motion_id'] not in byid;byid[meta['motion_id']]=meta
        assert meta['status']=='FINISHED_PHYSICAL_TRIAL' and meta['runtime_kind']=='isaaclab_physics'
        assert meta['phase']=='INITIAL_THEN_PRE_AUTORESET_TERMINATION_COMPUTE'
        assert meta['samples']==entry['samples']==meta['trial']['control_steps']+1
        assert sha(f/'poses.private.bin')==meta['binary_sha256'] and sha(f/'executed_poses.private.npz')==meta['npz_sha256']==entry['npz_sha256']
    assert set(byid)=={x['motion_id'] for x in log['motions']}
    for row in selected:
        meta=byid[source_id(row['reference'])];assert meta['trial']==row['trial']
        assert meta['reference_frames']==row['reference']['frames'] and meta['final_reference_endpoint_comparison_performed'] is False
    kept=[x for x in selected if x['include_in_candidate_view']]
    assert len(kept)==r['complete_trials'] and len(selected)-len(kept)==r['failed_trials']
    assert not (folder/'evidence_inputs.private.json').exists()
    bindings.update(private_strict=sha(strict),private_journal=sha(journal),private_execution_index=sha(index))
    summary=dict(status='QUICK_TRACKING_SUBSET_IDENTIFIED_CORPUS_NOT_READY' if kept else 'QUICK_TRACKING_SUBSET_EMPTY_NOT_TRAINABLE',
                 quality_label=QUALITY,examined_original_sources=64,nominal_tracking_completed_sources=len(kept),
                 failed_trial_sources=64-len(kept),unattempted_original_sources=8278-64,
                 retained_source_splits={s:sum(x['reference']['split']==s for x in kept) for s in ['train','val']},
                 mirror_physics_assessed=False,mirror_policy='originals_only; unassessed mirrors excluded',
                 final_reference_endpoint_verified=False,absolute_physical_infeasibility_claimed=False,
                 whole_library_filtered=False,dataset_accepted=False,training_ready=False,
                 source_artifact_rehash_and_pose_split_check_pending=True,normalization_and_clip_cache_pending=True,
                 actual_generator_training_started=False,source_assets_deleted=0,bindings=bindings)
    return selected,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--pilot-run',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    spec=json.loads(pathlib.Path(a.spec).read_text())
    assert spec['selection_enabled'] is True,'Quick corpus source selection disabled; no view or SDK created'
    assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run) and re.fullmatch('[a-z0-9_]+',a.pilot_run)
    assert sha(__file__)==spec['source_sha256'] and spec['mirror_policy']=='originals_only'
    selected,summary=select_actual(ROOT,a.pilot_run,spec)
    # Write only after every actual artifact/source identity check succeeds.
    assert ROOT.resolve()==ROOT and ROOT.stat().st_uid==1013
    parent=ROOT/'quick_source_views';parent.mkdir(mode=0o700,exist_ok=True)
    assert parent.resolve()==parent and parent.stat().st_uid==1013 and parent.stat().st_dev==ROOT.stat().st_dev
    out=parent/a.run;out.mkdir(mode=0o700,exist_ok=False)
    private=out/'source_decisions.private.json'
    with private.open('x') as stream:json.dump(selected,stream,indent=2)
    private.chmod(0o600)
    summary.update(source_sha256=sha(__file__),private_decisions_sha256=sha(private))
    with (out/'summary.json').open('x') as stream:json.dump(summary,stream,indent=2)
    print(json.dumps(summary))
if __name__=='__main__':main()
