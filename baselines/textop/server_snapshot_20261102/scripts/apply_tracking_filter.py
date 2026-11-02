"""Verify real two-stage evidence before any motion is accepted.

No boolean header, short PPO run, file check or ten-second timeout substitutes
for the actual40k policy and strict whole-motion rollouts. Protocolv2 supersedes
the earlier header-only schema; no accepted dataset existed under that schema.
Restricted per-motion evidence stays private, public receipts contain counts.
"""
import argparse,hashlib,json,math,pathlib,re
from strict_tracking_coverage import verify_trial

def artifact(evidence,key):
 x=evidence[key];p=pathlib.Path(x['path']);assert p.is_file(),key
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(8388608),b''):h.update(block)
 assert h.hexdigest()==x['sha256'],key
 return p

def apply(evidence):
 h=evidence['protocol'];assert h['schema']=='textop.dynamic_filter.v2'
 assert h['privileged_actor_and_critic'] is True and h['domain_randomization'] is False
 assert h['training_iterations']==40000 and h['num_envs']==8192
 assert h['relaxed_removed_terminations']==['anchor_pos','ee_body_pos']
 assert h['strict_standard_terminations_restored'] is True
 assert h['failure_average_definition']=='pinned_ema_failed_over_failed_plus_success_alpha0.001'
 repetitions=h['strict_rollouts_per_motion'];assert type(repetitions) is int and repetitions>=1
 assert h['strict_reset_start_frame']==0 and h['control_hz']==50
 assert h['strict_pelvis_height_threshold_m']==.25 and h['strict_end_effector_height_threshold_m']==.5
 paths={k:artifact(evidence,k) for k in ['policy_checkpoint','training_receipt','training_exit_receipt','relaxed_rollout_log','strict_rollout_log']}
 training=json.loads(paths['training_receipt'].read_text());exit_proof=json.loads(paths['training_exit_receipt'].read_text())
 assert training['status']=='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED' and training['mode']=='formal'
 assert training['formal_filter_training_started'] is True and training['num_envs']==8192
 assert training['completed_learning_iterations']==40000 and training['optimizer_updates']==800000
 assert training['latest_checkpoint']['sha256']==evidence['policy_checkpoint']['sha256']
 assert training['latest_checkpoint']['completed_learning_iterations']==40000
 assert training['latest_checkpoint']['actual_optimizer_updates']==800000
 assert exit_proof['child_exit_code']==0 and exit_proof['timed_out'] is False
 assert exit_proof['numerical_status']=='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED'
 fp=training['reference_fingerprint'];assert re.fullmatch('[0-9a-f]{64}',fp)
 relaxed=json.loads(paths['relaxed_rollout_log'].read_text());strict=json.loads(paths['strict_rollout_log'].read_text())
 for log in [relaxed,strict]:
  assert log['policy_checkpoint_sha256']==evidence['policy_checkpoint']['sha256'] and log['reference_fingerprint']==fp
 assert relaxed['status']=='RELAXED_TRAINED_POLICY_STATISTICS_EXPORTED'
 assert strict['status']=='STRICT_ENTIRE_MOTION_ROLLOUTS_FINISHED' and strict['diagnostic_only'] is False
 assert strict['standard_terminations_restored'] is True
 assert strict['pelvis_height_threshold_m']==.25 and strict['end_effector_height_threshold_m']==.5
 rows=relaxed['motions'];assert len(rows)==training['reference_sources'] and rows
 ids=[x['motion_id'] for x in rows];assert len(ids)==len(set(ids))
 assert all(re.fullmatch('[0-9a-f]{64}',x) for x in ids)
 eligible=[];decisions=[]
 for row in rows:
  assert type(row['raw_failed']) is int and type(row['raw_success']) is int
  assert row['raw_failed']>=0 and row['raw_success']>=0 and row['raw_failed']+row['raw_success']>0,'Never-attempted motions cannot pass'
  ef=row['ema_failed'];es=row['ema_success']
  assert all(type(x) in [float,int] and math.isfinite(x) and x>=0 for x in [ef,es]) and ef+es>0
  probability=ef/(ef+es);assert abs(probability-row['relaxed_failure_probability'])<1e-7
  assert type(row['reference_frames']) is int and row['reference_frames']>=2
  if probability>.05:decisions.append({'motion_id':row['motion_id'],'decision':'reject_relaxed_failure_rate'})
  else:eligible.append(row)
 strict_ids=[x['motion_id'] for x in strict['motions']]
 assert len(strict_ids)==len(set(strict_ids)) and set(strict_ids)=={x['motion_id'] for x in eligible}
 strict_by_id={x['motion_id']:x for x in strict['motions']}
 for row in eligible:
  trials=strict_by_id[row['motion_id']]['trials'];assert len(trials)==repetitions
  failures=sum(verify_trial(trial,row['reference_frames'],i) for i,trial in enumerate(trials))
  decisions.append({'motion_id':row['motion_id'],'decision':'reject_strict_failure' if failures else 'retain'})
 assert len(decisions)==len(rows)
 return decisions

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve existing filter evidence')
 decisions=apply(json.loads(a.evidence.read_text()))
 a.output.write_text(json.dumps({'decisions':decisions,'evidence_file':str(a.evidence)},indent=2))
