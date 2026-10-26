"""New pilot provenance/terminal-trace CPU fixtures, never SDK or real acceptance."""
import argparse, copy, hashlib, json, pathlib, sys, time
import numpy as np
BASE=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'scripts'))
from quick_pilot_protocol import gate, authorize_teacher, select_sources, settings_signature, QUALITY
from quick_budget_guards import RUN, OUT, PLAN_SHA
from quick_pilot_trace import ExecutionTrace, WIDTH
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
spec=json.loads((BASE/'configs/quick-pilot-spec.json').read_text(encoding='utf-8'))
checks=[]
def rejects(name, callback):
 try: callback()
 except (AssertionError,KeyError,ValueError,FileExistsError):checks.append(name);return
 raise AssertionError('Expected refusal: '+name)
def passes(name, condition):
 assert condition,name
 checks.append(name)

# A completely synthetic chain, not copied into any real stop/checkpoint path.
cp=dict(path=OUT+'/completed_5000.pt',sha256='a'*64,bytes=90000000)
digests={k:hashlib.sha256(('SYNTHETIC_'+k).encode()).hexdigest() for k in ['training','stop','backup','quick_plan','original_exit']}
ss=copy.deepcopy(spec);ss['quick_plan_sha256']=digests['quick_plan']
ex=dict(child_pid=111,timed_out=False,numerical_status='FAILED',child_exit_code=1)
backup=dict(status='FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED',completed=5000,
 audit=dict(completed_learning_iterations=5000,actual_optimizer_updates=100000,all_finite_tensor_count=80,
            optimizer_step_min=100000,optimizer_step_max=100000,plan_sha256=PLAN_SHA,
            reference_fingerprint=spec['expected_reference_fingerprint'],full_checkpoint=copy.deepcopy(cp)),
 private_full_two_location_sha_verified=True,inference_three_location_sha_verified=True,
 original_server_checkpoint_retained=True,data_deleted=False,full_checkpoint_publicly_uploaded=False)
qp=dict(requested_run=RUN,quality_label=QUALITY,quick_target_iterations=5000,paper_target_iterations=40000,
        early_stop_enabled=True,user_request='SYNTHETIC user quick-budget request')
training=dict(mode='formal',status='FAILED',plan_sha256=PLAN_SHA,num_envs=8192,
 completed_learning_iterations=5001,optimizer_updates=100023,reference_sources=8278,
 reference_fingerprint=spec['expected_reference_fingerprint'],
 latest_checkpoint=dict(name='completed_5000.pt',completed_learning_iterations=5000,actual_optimizer_updates=100000,sha256='a'*64,bytes=90000000))
stop=dict(status='USER_AUTHORIZED_QUICK_5000_EARLY_STOP_COMPLETE',action='user_authorized_early_stop_after_backup',
 quality_label=QUALITY,user_request=qp['user_request'],full_paper_training_completed=False,checkpoint_retained=True,
 quick_checkpoint_completed_iterations=5000,signal_identity='pidfd_original_process',trainer_start_ticks=123,
 signals_sent=[dict(pid=222,signal='SIGINT')],trainer_pid=222,owned_group=111,original_supervisor_exit=copy.deepcopy(ex),
 original_final_receipt_sha256=digests['training'],original_exit_receipt_sha256=digests['original_exit'],
 backup_receipt_sha256=digests['backup'],plan_sha256=digests['quick_plan'],checkpoint=copy.deepcopy(cp))
teacher=dict(training=training,stop=stop,backup=backup,quick_plan=qp,original_exit=ex,artifact_sha256=digests)
passes('controlled5k_with_partial_post_checkpoint_updates_is_allowed_as_pilot_only',authorize_teacher(ss,teacher)==cp)
for name,group,key,value in [
 ('missing_actual_controlled_exit','stop','status','QUICK_STOP_PENDING'),
 ('no_verified_private_backup','backup','private_full_two_location_sha_verified',False),
 ('no_verified_public_inference_backup','backup','inference_three_location_sha_verified',False),
 ('full_recovery_file_publication_refused','backup','full_checkpoint_publicly_uploaded',True),
 ('numeric_pid_signal_binding_refused','stop','signal_identity','numeric_pid'),
 ('wrong_5k_optimization_count_refused','training','latest_checkpoint',dict(training['latest_checkpoint'],actual_optimizer_updates=20000)),
 ('40k_completion_not_a_quick_stop_fixture','training','completed_learning_iterations',40000),
 ('mismatched_source_order_refused','training','reference_fingerprint','b'*64),
 ('edited_final_receipt_refused','stop','original_final_receipt_sha256','c'*64),
 ('deleted_original_checkpoint_refused','stop','checkpoint_retained',False)]:
 t=copy.deepcopy(teacher);t[group][key]=value
 rejects(name,lambda t=t:authorize_teacher(ss,t))
rejects('both_pilot_stages_initially_disabled',lambda:gate('smoke',spec))
enabled=copy.deepcopy(spec);enabled['pilot_launch_enabled']=True
rejects('representative_stage_separately_disabled',lambda:gate('sample',enabled))
rows=[dict(canonical_source=g+'/motion_'+str(i),frames=n) for g in ['A','B','C'] for n in [60,400,1200] for i in range(10)]
original=copy.deepcopy(rows)
short=select_sources(rows,'smoke',spec);sample=select_sources(rows,'sample',spec)
passes('smoke_uses_eight_whole_shortest_sources',len(short)==8 and all(rows[i]['frames']==60 for i in short))
passes('representative_selection_is_unique_and_dataset_duration_stratified',len(set(sample))==64 and len({(rows[i]['canonical_source'][0],rows[i]['frames']) for i in sample})==9)
passes('selection_is_deterministic_and_does_not_mutate_input',sample==select_sources(rows,'sample',spec) and rows==original)
rows_with_failure=copy.deepcopy(rows)
for i,row in enumerate(rows_with_failure):row['failed_motion_count']=i*100
passes('early_teacher_failure_statistics_do_not_affect_selection',sample==select_sources(rows_with_failure,'sample',spec))
changed=copy.deepcopy(spec);changed['strict_pelvis_height_threshold_m']=.3
passes('changed_threshold_changes_execution_signature',settings_signature(changed)!=settings_signature(spec))

def state(value,frame=0):
 return dict(reference_frame=np.array([frame],dtype=np.float32),root_pos=np.ones(3,dtype=np.float32)*value,
 root_quat_wxyz=np.array([1,0,0,0],dtype=np.float32),joint_pos=np.ones(29,dtype=np.float32)*value,
 reference_joint_pos=np.zeros(29,dtype=np.float32),actions=np.zeros(29,dtype=np.float32),
 body_pos=np.ones((14,3),dtype=np.float32)*value,reference_body_pos=np.zeros((14,3),dtype=np.float32))
names=['joint_'+str(i) for i in range(29)];tr=ExecutionTrace(out/'synthetic_private'/'terminal', 'a'*64,3,names,runtime_kind='synthetic_cpu_fixture')
tr.append(state(0));terminal=state(.2);tr.append(terminal)
for k in terminal:terminal[k].fill(999)
tr.append(state(.4,frame=1))
trial=dict(repeat=0,start_frame=0,reference_frames=3,control_steps=2,end_reason='motion_end',failed=False,complete=True)
record=tr.finish(trial)
with np.load(tr.folder/'executed_poses.private.npz',allow_pickle=False) as z:
 passes('terminal_pose_is_copied_before_autoreset_mutates_input',np.all(z['joint_pos'][1]==np.float32(.2)))
 passes('trace_preserves_initial_and_all_physical_transitions',z['joint_pos'].shape==(3,29) and np.allclose(z['time_seconds'],[0,.02,.04]))
 passes('synthetic_trace_cannot_be_labeled_actual_simulation',str(z['runtime_kind'])=='synthetic_cpu_fixture' and tr.meta['status']=='FINISHED_SYNTHETIC_FIXTURE')
passes('binary_journal_retained_and_hashed',tr.path.stat().st_size==3*WIDTH*4 and tr.meta['binary_sha256']==hashlib.sha256(tr.path.read_bytes()).hexdigest())
passes('consumed_target_range_is_explicit_without_claiming_final_endpoint',tr.meta['consumed_reference_frame_max']==1 and tr.meta['reference_final_frame']==2 and tr.meta['final_reference_endpoint_comparison_performed'] is False)
rejects('trace_path_reuse_refused',lambda:ExecutionTrace(tr.folder,'a'*64,3,names))
partial=ExecutionTrace(out/'synthetic_private'/'partial','b'*64,3,names,runtime_kind='synthetic_cpu_fixture');partial.append(state(0))
rejects('truncated_pose_trace_cannot_finish_as_whole_motion',lambda:partial.finish(trial))
partial.close_incomplete()
passes('incomplete_journal_survives_without_success_claim',json.loads((partial.folder/'trace.private.json').read_text())['status']=='INCOMPLETE' and partial.path.stat().st_size==WIDTH*4)
bad=ExecutionTrace(out/'synthetic_private'/'nonfinite','c'*64,3,names,runtime_kind='synthetic_cpu_fixture');st=state(0);st['joint_pos'][0]=np.nan
rejects('nonfinite_state_refused_before_journal_write',lambda:bad.append(st));bad.close_incomplete()
passes('bad_state_does_not_grow_binary_journal',bad.path.stat().st_size==0)
tiny=ExecutionTrace(out/'synthetic_private'/'capacity','d'*64,3,names,runtime_kind='synthetic_cpu_fixture',budget={'used':0,'maximum':4})
rejects('trace_storage_budget_checked_before_write',lambda:tiny.append(state(0)));tiny.close_incomplete()
jump=ExecutionTrace(out/'synthetic_private'/'reference_jump','e'*64,3,names,runtime_kind='synthetic_cpu_fixture');jump.append(state(0))
rejects('autoreset_or_target_jump_refused_inside_trial',lambda:jump.append(state(.1,frame=2)));jump.close_incomplete()
fail=ExecutionTrace(out/'synthetic_private'/'failure_trial','f'*64,3,names,runtime_kind='synthetic_cpu_fixture');fail.append(state(0));fail.append(state(.3))
failed_trial=dict(trial,control_steps=1,end_reason='termination',failed=True,complete=False)
failed_record=fail.finish(failed_trial)
passes('actual_failure_record_not_converted_to_natural_completion',failed_record['physical_failure'] is True and fail.meta['trial']['complete'] is False)
result=dict(status='QUICK_PILOT_NEW_CPU_FIXTURES_PASSED',checks=checks,check_count=len(checks),
 sdk_runs=0,new_optimizer_updates=0,training_signals_sent=0,real_motion_trials=0,dataset_accepted=False,
 synthetic_fixture_only=True,quality_label=QUALITY,source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),sample_unix=time.time())
(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
