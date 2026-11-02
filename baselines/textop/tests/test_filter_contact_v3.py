"""Synthetic post-exit/contact evidence checks: zero physics, zero motion acceptance."""
import argparse,copy,hashlib,json,pathlib,sys,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from apply_tracking_filter_v3 import apply,contact_signature,require_completed_training

class Fixture:
 def __init__(self,root):
  self.root=root;root.mkdir(exist_ok=False)
  self.e={'protocol':{'schema':'textop.dynamic_filter.v3','privileged_actor_and_critic':True,'domain_randomization':False,'training_iterations':40000,'num_envs':8192,'relaxed_removed_terminations':['anchor_pos','ee_body_pos'],'strict_standard_terminations_restored':True,'failure_average_definition':'pinned_ema_failed_over_failed_plus_success_alpha0.001','strict_rollouts_per_motion':1,'strict_reset_start_frame':0,'control_hz':50,'strict_pelvis_height_threshold_m':.25,'strict_end_effector_height_threshold_m':.5}}
  self.write('policy_checkpoint',b'SYNTHETIC_SCHEMA_ONLY_NOT_A_TRAINED_POLICY',True)
  self.write('evaluation_source',b'SYNTHETIC_SOURCE_MARKER_NOT_AN_EXECUTOR',True)
  cp=self.e['policy_checkpoint']['sha256'];source=self.e['evaluation_source']['sha256'];fp='f'*64
  self.profile={'kind':'validated_static_solid_floor','top_z':0.,'dimensions_m':[4000,4000,.2],'static_friction':1.,'dynamic_friction':1.,'restitution':0.,'combine_mode':'multiply','self_collisions':True,'data_or_reward_changes':False,'validation_witnesses':[{'run':x,'receipt_sha256':'1'*64,'actual_iterations':10,'actual_optimizer_updates':200} for x in ['synthetic_author','synthetic_document']]}
  signature=contact_signature(self.profile)
  self.plan={'contact_profile':{'enabled':True,'kind':'validated_static_solid_floor'}}
  self.spec={'executor_sha256':source,'formal_evaluation_enabled':True,'real_sdk_validation_completed':True,'acceptance_schema':'textop.dynamic_filter.v3','strict_rollouts_per_motion':1}
  self.train={'status':'TEXTOP_FORMAL_FILTER_TRAINING_FINISHED','mode':'formal','formal_filter_training_started':True,'num_envs':8192,'completed_learning_iterations':40000,'optimizer_updates':800000,'reference_sources':1,'reference_fingerprint':fp,'latest_checkpoint':{'sha256':cp,'completed_learning_iterations':40000,'actual_optimizer_updates':800000},'output_writers_closed':True,'contact_profile':copy.deepcopy(self.profile)}
  self.train_exit={'child_exit_code':0,'timed_out':False,'numerical_status':self.train['status']}
  self.relaxed={'status':'RELAXED_TRAINED_POLICY_STATISTICS_EXPORTED','policy_checkpoint_sha256':cp,'reference_fingerprint':fp,'contact_profile_sha256':signature,'motions':[{'motion_id':'a'*64,'raw_failed':0,'raw_success':10,'ema_failed':0.,'ema_success':1.,'relaxed_failure_probability':0.,'reference_frames':4}]}
  self.strict={'status':'STRICT_ENTIRE_MOTION_ROLLOUTS_FINISHED','diagnostic_only':False,'policy_checkpoint_sha256':cp,'reference_fingerprint':fp,'contact_profile_sha256':signature,'standard_terminations_restored':True,'pelvis_height_threshold_m':.25,'end_effector_height_threshold_m':.5,'motions':[{'motion_id':'a'*64,'trials':[{'repeat':0,'start_frame':0,'reference_frames':4,'control_steps':3,'end_reason':'motion_end','complete':True,'failed':False}]}]}
  self.evaluation={'status':'TEXTOP_TWO_ROUND_FILTER_EXPORT_FINISHED','mode':'formal','output_writers_closed':True,'physical_filter_executed':True,'dataset_accepted':False,'optimizer_updates':0,'diagnostic_dynamic_rollouts_executed':False,'source_sha256':source,'policy_checkpoint_sha256':cp,'policy_checkpoint_completed_iterations':40000,'reference_fingerprint':fp,'reference_sources':1,'contact_profile':copy.deepcopy(self.profile),'contact_profile_sha256':signature,'eligible_original_sources':1,'finished_trials':1,'failed_trials':0,'complete_trials':1,'actual_scene_physics_steps':12,'actual_vectorized_control_calls':3}
  self.journal=[{'motion_id':row['motion_id'],'trial':trial} for row in self.strict['motions'] for trial in row['trials']]
  self.eval_exit={'child_exit_code':0,'timed_out':False,'numerical_status':self.evaluation['status']}
  self.process={'child_pid':12345,'child_exit_code':0,'child_reaped':True,'timed_out':False,'supervision_passed':True,'supervisor_error':None,'sample_errors':[],'samples_written':3}
  self.samples=[{'phase':x,'synthetic_resource_sample':True} for x in ['before_launch','after_launch','after_exit']]
  self.eval_exit.update(self.process)
  self.refresh()
 def write(self,key,data,binary=False):
  path=self.root/key;path.write_bytes(data if binary else json.dumps(data).encode())
  self.e[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 def refresh(self):
  self.write('training_plan',self.plan);self.write('evaluation_spec',self.spec)
  self.train['plan_sha256']=self.e['training_plan']['sha256']
  self.write('training_receipt',self.train);self.write('training_exit_receipt',self.train_exit)
  self.write('relaxed_rollout_log',self.relaxed);self.write('strict_rollout_log',self.strict)
  self.write('strict_trial_journal',('\n'.join(json.dumps(x) for x in self.journal)+'\n').encode(),True)
  self.evaluation.update(training_receipt_sha256=self.e['training_receipt']['sha256'],training_plan_sha256=self.e['training_plan']['sha256'],replay_spec_sha256=self.e['evaluation_spec']['sha256'],private_relaxed_log_sha256=self.e['relaxed_rollout_log']['sha256'],private_strict_log_sha256=self.e['strict_rollout_log']['sha256'],private_trial_journal_sha256=self.e['strict_trial_journal']['sha256'])
  self.write('evaluation_receipt',self.evaluation)
  self.write('process_exit_receipt',self.process)
  self.write('resource_samples',('\n'.join(json.dumps(x) for x in self.samples)+'\n').encode(),True)
  self.eval_exit['process_exit_sha256']=self.e['process_exit_receipt']['sha256']
  self.eval_exit['resources_sha256']=self.e['resource_samples']['sha256']
  self.eval_exit['evaluation_receipt_sha256']=self.e['evaluation_receipt']['sha256'];self.write('evaluation_exit_receipt',self.eval_exit)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();root=a.output.absolute()
 assert not root.exists();root.mkdir(parents=False);assert root==root.resolve()
 checks=[]
 def case(name,mutate=None,expected_failure=True):
  f=Fixture(root/name)
  if mutate:mutate(f)
  f.refresh()
  try:result=apply(f.e)
  except (AssertionError,KeyError,ValueError) as e:
   assert expected_failure,(name,str(e));checks.append({'name':name,'expected_rejection_observed':True,'reason':str(e)})
  else:
   assert not expected_failure,name
   assert result[0]['decision']=='retain';checks.append({'name':name,'synthetic_schema_consistent':True,'real_motions_accepted':0})
 case('synthetic_consistency_only',expected_failure=False)
 case('running_policy_rejected',lambda f:f.train.update(status='RUNNING_FORMAL_FILTER_TRAINING',completed_learning_iterations=100,optimizer_updates=2000))
 case('unclosed_training_rejected',lambda f:f.train.update(output_writers_closed=False))
 case('old_plane_profile_rejected',lambda f:f.evaluation['contact_profile'].update(kind='legacy_plane'))
 case('selfcollision_disabled_rejected',lambda f:f.evaluation['contact_profile'].update(self_collisions=False))
 case('friction_changed_rejected',lambda f:f.evaluation['contact_profile'].update(dynamic_friction=.5))
 case('log_contact_hash_mismatch',lambda f:f.strict.update(contact_profile_sha256='0'*64))
 case('failed_evaluator_exit',lambda f:f.eval_exit.update(child_exit_code=1))
 case('timeout_evaluator_exit',lambda f:f.eval_exit.update(timed_out=True))
 case('unclosed_evaluator_rejected',lambda f:f.evaluation.update(output_writers_closed=False))
 case('diagnostic_evaluator_rejected',lambda f:f.evaluation.update(diagnostic_dynamic_rollouts_executed=True))
 case('evaluator_took_updates_rejected',lambda f:f.evaluation.update(optimizer_updates=20))
 case('wrong_executor_source',lambda f:f.evaluation.update(source_sha256='0'*64))
 case('disabled_evaluation_spec',lambda f:f.spec.update(formal_evaluation_enabled=False))
 case('unvalidated_evaluation_spec',lambda f:f.spec.update(real_sdk_validation_completed=False))
 case('wrong_evaluated_policy',lambda f:f.evaluation.update(policy_checkpoint_sha256='0'*64))
 case('wrong_reference_order',lambda f:f.evaluation.update(reference_fingerprint='0'*64))
 case('wrong_trial_count',lambda f:f.evaluation.update(finished_trials=2))
 case('wrong_physics_step_count',lambda f:f.evaluation.update(actual_scene_physics_steps=3))
 case('missing_journal_trial',lambda f:f.journal.clear())
 case('duplicated_journal_trial',lambda f:f.journal.append(copy.deepcopy(f.journal[0])))
 case('partial_trajectory_still_rejected',lambda f:(f.strict['motions'][0]['trials'][0].update(control_steps=1),f.journal[0]['trial'].update(control_steps=1)))
 case('unattempted_source_still_rejected',lambda f:f.relaxed['motions'][0].update(raw_success=0))
 case('supervisor_failure_even_child_exit0',lambda f:f.eval_exit.update(supervision_passed=False))
 case('process_exit_disagrees_with_wrapper',lambda f:f.process.update(child_exit_code=7))
 case('sampling_error_retained',lambda f:f.process.update(sample_errors=[{'error':'synthetic'}]))
 case('missing_after_exit_sample',lambda f:f.samples.pop())
 receipt={'status':'CONTACT_FILTER_V3_SYNTHETIC_PROTOCOL_CHECKS_PASSED','checks':checks,'checks_passed':len(checks),'gpu_calls':0,'physics_calls':0,'optimizer_updates':0,'physical_filter_executed':False,'dataset_accepted':False,'synthetic_fixture_markers_retained':True,'fixtures_removed':False,'elapsed_end_unix':time.time()}
 (root/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='checks'}))
if __name__=='__main__':main()
