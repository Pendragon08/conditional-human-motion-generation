"""New source-ID/coverage selection tests; synthetic metadata only, no SDK."""
import argparse, copy, hashlib, json, pathlib, sys
B=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'scripts'))
from select_quick_corpus_sources import decisions, source_id
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False);checks=[]
rows=[dict(canonical_source='synthetic_dataset/motion_'+str(i),split='train' if i<2 else 'val',
           mirrored=False,frames=40+i,physical_filter_passed=False) for i in range(4)]
def trial(frames,failed=False):
 return dict(repeat=0,start_frame=0,reference_frames=frames,control_steps=3 if failed else frames-1,
             end_reason='termination' if failed else 'motion_end',failed=failed,complete=not failed)
motions=[dict(motion_id=source_id(row),trials=[trial(row['frames'],i%2==1)]) for i,row in enumerate(rows)]
original=copy.deepcopy(rows)
def rejected(name,callback):
 try:callback()
 except (AssertionError,KeyError):checks.append(name);return
 raise AssertionError('Expected refusal: '+name)
def check(name,condition):
 assert condition,name
 checks.append(name)
result=decisions(rows,list(range(4)),motions)
check('success_failure_and_source_split_mapping_preserved',sum(x['include_in_candidate_view'] for x in result)==2 and [x['reference']['split'] for x in result]==['train','train','val','val'])
check('original_metadata_unchanged_and_copied',rows==original and all(x['reference'] is not rows[i] for i,x in enumerate(result)))
check('nominal_completion_never_promoted_to_training_ready_or_full_physics',all(x['dataset_accepted'] is False and x['training_ready'] is False and x['final_reference_endpoint_verified'] is False and x['reference']['physical_filter_passed'] is False for x in result))
check('failed_trial_not_absolute_infeasibility_or_asset_deletion',all(x['absolute_physical_infeasibility_claimed'] is False for x in result) and result[1]['failed_in_this_teacher_trial'] is True)
x=copy.deepcopy(motions);x[0]['trials'][0]=trial(2)
rejected('forged_trial_self_length_cannot_replace_known_original_length',lambda:decisions(rows,[0,1,2,3],x))
x=copy.deepcopy(motions);x[0]['trials'][0]['control_steps']-=1
rejected('truncated_natural_completion_refused',lambda:decisions(rows,[0,1,2,3],x))
x=copy.deepcopy(motions);x[0]['motion_id']='0'*64
rejected('foreign_source_identity_refused',lambda:decisions(rows,[0,1,2,3],x))
x=copy.deepcopy(motions);x[1]['motion_id']=x[0]['motion_id']
rejected('duplicate_source_trial_refused',lambda:decisions(rows,[0,1,2,3],x))
rejected('missing_selected_source_refused',lambda:decisions(rows,[0,1,2,3],motions[:-1]))
x=copy.deepcopy(rows);x[0]['mirrored']=True
rejected('unassessed_mirror_cannot_masquerade_as_assessed_original',lambda:decisions(x,[0,1,2,3],motions))
x=copy.deepcopy(motions);x[0]['trials'][0].update(end_reason='external_timeout',complete=False)
rejected('external_timeout_not_a_completed_physical_trial',lambda:decisions(rows,[0,1,2,3],x))
failed=[dict(motion_id=source_id(row),trials=[trial(row['frames'],True)]) for row in rows]
check('all_failed_trials_leave_no_retained_candidate',not any(x['include_in_candidate_view'] for x in decisions(rows,[0,1,2,3],failed)))
rejected('duplicate_requested_source_indices_refused',lambda:decisions(rows,[0,0,2,3],motions))
x=copy.deepcopy(rows);x[0]['split']='test_unregistered'
rejected('unexpected_source_split_refused',lambda:decisions(x,[0,1,2,3],motions))
x=copy.deepcopy(motions);x[0]['trials'].append(trial(rows[0]['frames']))
rejected('cannot_select_best_of_multiple_trials_silently',lambda:decisions(rows,[0,1,2,3],x))
assert not {'torch','mujoco','isaaclab'}.intersection(sys.modules)
summary=dict(status='QUICK_SOURCE_DECISION_SYNTHETIC_CPU_CHECKS_PASSED',check_count=len(checks),checks=checks,
             source_sha256=hashlib.sha256((B/'scripts/select_quick_corpus_sources.py').read_bytes()).hexdigest(),
             fixture_original_sources=4,real_pilot_sources_examined=0,actual_source_retention_decisions=0,
             real_corpus_created=False,dataset_accepted=False,sdk_runs=0,new_optimizer_updates=0,training_signals_sent=0)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary))
