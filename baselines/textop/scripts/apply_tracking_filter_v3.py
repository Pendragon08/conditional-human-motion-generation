"""Post-exit filter evidence with identical training/evaluation contact physics.

This validates consistency and provenance records, not their independent truth.
Only a trusted actually executed exporter/supervisor can establish real rollouts.
No GPU work or dataset changes occur here. Per-motion decisions remain private.
"""
import argparse,copy,hashlib,json,pathlib
from apply_tracking_filter import artifact,apply as apply_v2


def contact_signature(profile):
 assert profile['kind']=='validated_static_solid_floor'
 assert profile['top_z']==0.0 and profile['dimensions_m']==[4000,4000,0.2]
 assert profile['static_friction']==profile['dynamic_friction']==1.0
 assert profile['restitution']==0.0 and profile['combine_mode']=='multiply'
 assert profile['self_collisions'] is True and profile['data_or_reward_changes'] is False
 witnesses=profile['validation_witnesses']
 assert len(witnesses)==2 and len({x['run'] for x in witnesses})==2
 assert all(x['actual_iterations']==10 and x['actual_optimizer_updates']==200 for x in witnesses)
 return hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require_completed_training(training):
 assert training['status']=='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED','Actual40k training is incomplete'
 assert training['mode']=='formal' and training['formal_filter_training_started'] is True
 assert training['completed_learning_iterations']==40000 and training['optimizer_updates']==800000
 assert training['num_envs']==8192 and training['output_writers_closed'] is True


def apply(evidence):
 assert evidence['protocol']['schema']=='textop.dynamic_filter.v3'
 training=json.loads(artifact(evidence,'training_receipt').read_text())
 require_completed_training(training)
 plan_path=artifact(evidence,'training_plan');plan=json.loads(plan_path.read_text())
 assert evidence['training_plan']['sha256']==training['plan_sha256']
 assert plan['contact_profile']['enabled'] is True
 assert plan['contact_profile']['kind']==training['contact_profile']['kind']
 profile_sha=contact_signature(training['contact_profile'])
 paths={k:artifact(evidence,k) for k in ['evaluation_receipt','evaluation_exit_receipt','evaluation_source','evaluation_spec','relaxed_rollout_log','strict_rollout_log','strict_trial_journal']}
 evaluation=json.loads(paths['evaluation_receipt'].read_text())
 exit_proof=json.loads(paths['evaluation_exit_receipt'].read_text())
 spec=json.loads(paths['evaluation_spec'].read_text())
 assert evaluation['status']=='TEXTOP_TWO_ROUND_FILTER_EXPORT_FINISHED'
 assert evaluation['mode']=='formal' and evaluation['output_writers_closed'] is True
 assert evaluation['physical_filter_executed'] is True and evaluation['dataset_accepted'] is False
 assert evaluation['optimizer_updates']==0 and evaluation['diagnostic_dynamic_rollouts_executed'] is False
 assert exit_proof['child_exit_code']==0 and exit_proof['timed_out'] is False
 assert exit_proof['numerical_status']==evaluation['status']
 assert exit_proof['evaluation_receipt_sha256']==evidence['evaluation_receipt']['sha256']
 assert evaluation['source_sha256']==evidence['evaluation_source']['sha256']==spec['executor_sha256']
 assert evaluation['replay_spec_sha256']==evidence['evaluation_spec']['sha256']
 assert spec['formal_evaluation_enabled'] is True and spec['real_sdk_validation_completed'] is True and spec['acceptance_schema']=='textop.dynamic_filter.v3'
 assert evaluation['training_receipt_sha256']==evidence['training_receipt']['sha256']
 assert evaluation['training_plan_sha256']==evidence['training_plan']['sha256']
 assert evaluation['policy_checkpoint_sha256']==evidence['policy_checkpoint']['sha256']
 assert evaluation['policy_checkpoint_completed_iterations']==40000
 assert evaluation['reference_fingerprint']==training['reference_fingerprint']
 assert evaluation['reference_sources']==training['reference_sources']
 assert contact_signature(evaluation['contact_profile'])==profile_sha==evaluation['contact_profile_sha256']
 relaxed=json.loads(paths['relaxed_rollout_log'].read_text());strict=json.loads(paths['strict_rollout_log'].read_text())
 assert relaxed['contact_profile_sha256']==strict['contact_profile_sha256']==profile_sha
 assert evaluation['private_relaxed_log_sha256']==evidence['relaxed_rollout_log']['sha256']
 assert evaluation['private_strict_log_sha256']==evidence['strict_rollout_log']['sha256']
 repetitions=evidence['protocol']['strict_rollouts_per_motion']
 assert spec['strict_rollouts_per_motion']==repetitions
 trials=[t for row in strict['motions'] for t in row['trials']]
 journal=[json.loads(line) for line in paths['strict_trial_journal'].read_text().splitlines()]
 expected_journal=[{'motion_id':row['motion_id'],'trial':trial} for row in strict['motions'] for trial in row['trials']]
 encode=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
 assert sorted(map(encode,journal))==sorted(map(encode,expected_journal)),'Lost or duplicated trial journal'
 assert evaluation['private_trial_journal_sha256']==evidence['strict_trial_journal']['sha256']
 assert evaluation['eligible_original_sources']==len(strict['motions'])
 assert evaluation['finished_trials']==len(trials)==len(strict['motions'])*repetitions
 assert evaluation['failed_trials']==sum(t['failed'] for t in trials)
 assert evaluation['complete_trials']==sum(t['complete'] for t in trials)
 assert evaluation['actual_scene_physics_steps']==4*evaluation['actual_vectorized_control_calls']
 if trials:assert evaluation['actual_vectorized_control_calls']>0
 # Retain all v2 training/checkpoint/attempt/failure/continuous-coverage checks.
 legacy=copy.deepcopy(evidence);legacy['protocol']['schema']='textop.dynamic_filter.v2'
 return apply_v2(legacy)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 assert not a.output.exists(),'Never replace prior decisions'
 decisions=apply(json.loads(a.evidence.read_text()))
 a.output.write_text(json.dumps({'schema':'textop.dynamic_filter.v3','decisions':decisions,'dataset_accepted':False,'corpus_acceptance_pending':True},indent=2))
