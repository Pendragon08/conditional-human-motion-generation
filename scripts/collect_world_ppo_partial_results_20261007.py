"""Read and validate completed stages only; never tune or change the live queue."""
from pathlib import Path
import json,hashlib,math
from datetime import datetime
from zoneinfo import ZoneInfo
from r3_world_ssh_transport import ROOT,REMOTE,execute
program=r'''
from pathlib import Path
import json,hashlib,time
r=Path(REMOTE);state=json.loads((r/'artifacts/monitoring/r3_world_ppo_formal_20261007.state').read_text());records=[]
for item in state.get('completed',[]):
 p=Path(item['output']);data=json.loads((p/'summary.json').read_text())
 pref=r/f"artifacts/monitoring/r3_world_ppo_{item['stage']}_{item['arm']}_{item['seed']}_20261007"
 record={**item,'summary':data,'summary_sha256_actual':hashlib.sha256((p/'summary.json').read_bytes()).hexdigest(),
  'stage_state':pref.with_suffix('.state').read_text()}
 marker='WORLD_RESIDUAL_PPO_CONTROL_COMPLETE' if item['stage']=='train' else 'WORLD_PPO_CONTROL_EVALUATION_COMPLETE'
 record['completion_sentinel_count']=pref.with_suffix('.log').read_text(errors='replace').count(marker)
 if item['stage']=='train':
  ckpt=p/'model_update0500.pt';record['final_checkpoint_sha256']=hashlib.sha256(ckpt.read_bytes()).hexdigest();record['final_checkpoint_bytes']=ckpt.stat().st_size
  rows=[json.loads(x) for x in (p/'updates.jsonl').read_text().splitlines()];record['update_rows']=rows
 records.append(record)
print(json.dumps({'observed_unix':time.time(),'queue':state,'completed_records':records}))
'''.replace('REMOTE',repr(REMOTE))
data=json.loads(execute(program));base=ROOT/'results/world_residual_ppo_20261007';raw=base/'raw/monitoring'/f"completed_verified_{int(data['observed_unix'])}.json"
raw.parent.mkdir(parents=True,exist_ok=True);raw.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def finite(value):
 if isinstance(value,float):return math.isfinite(value)
 if isinstance(value,dict):return all(finite(v) for v in value.values())
 if isinstance(value,list):return all(finite(v) for v in value)
 return True
train={};rows=[]
for record in data['completed_records']:
 s=record['summary'];key=(record['arm'],record['seed'])
 assert record['summary_sha256']==record['summary_sha256_actual']
 assert record['stage_state'].startswith('COMPLETE ') and 'child_exit=0' in record['stage_state']
 assert record['completion_sentinel_count']==1 and s['status']=='COMPLETE' and s['mode']=='formal'
 if record['stage']=='train':
  assert s['formal_PPO_updates']==500 and s['actual_transitions']==98304000
  u=record['update_rows'];assert len(u)==500 and finite(u)
  assert [x['completed_update'] for x in u]==list(range(1,501))
  assert all(x['actual_transitions']==x['completed_update']*8192*24 for x in u)
  assert s['norm_frozen'] and s['no_extra_IK'] and s['imagined_transitions']==0
  assert s['restored_optimizer'] and s['original_task_reward_PD_termination_unchanged']
  assert s['aux_updates']==(500 if record['arm'].startswith('W-') else 0)
  train[key]=record
 else:
  assert key in train and s['checkpoint_completed_updates']==500
  assert s['checkpoint_sha256']==train[key]['final_checkpoint_sha256']
  assert s['actual_evaluation_transitions']==1024000 and s['num_envs']==2048 and s['steps']==500
  assert s['PPO_updates']==s['aux_updates']==s['normalizer_updates']==0
  assert s['model_hashes_unchanged'] and len(s['rows'])==11 and finite(s['rows'])
  fields={'first_episode_macro_success':'first_episode_success','first_episode_macro_failure':'first_episode_failure',
   'recovery_macro_joint_RMSE_AUC':'recovery_joint_RMSE_AUC_rad_s','clean_macro_recovery_RMSE':'clean_recovery_RMSE_rad'}
  for macro,field in fields.items():assert abs(s[macro]-sum(x[field] for x in s['rows'])/11)<1e-12,macro
  rows.append({'arm':key[0],'seed':key[1],'PPO_updates':500,
   'first_episode_success':s['first_episode_macro_success'],'first_episode_failure':s['first_episode_macro_failure'],
   'recovery_RMSE_AUC_rad_s':s['recovery_macro_joint_RMSE_AUC'],'clean_recovery_RMSE_rad':s['clean_macro_recovery_RMSE'],
   'official_any_reset_success_rate':s['official_any_reset_success_rate'],'checkpoint_sha256':s['checkpoint_sha256'],
   'evaluation_summary_sha256':record['summary_sha256'],'evaluation_video_verified':s['evaluation_video_verified']})
report={'status':'PARTIAL_NO_FINAL_GATE_DECISION','observed_unix':data['observed_unix'],
 'observed_beijing':datetime.fromtimestamp(data['observed_unix'],ZoneInfo('Asia/Shanghai')).isoformat(),
 'completed_train_runs':len(train),'completed_eval_runs':len(rows),'planned_runs_each':15,'rows':rows,
 'completed_summary_state_checkpoint_identity_and_scalar_macro_checks_passed':True,
 'full_trace_independent_audit_performed_this_monitoring_turn':False,
 'final_control_gate_evaluated':False,'innovation_validated':False,'same_config_as_old_85_3_percent':False,
 'zero_task_success_not_process_failure':True,'no_queue_or_scientific_source_mutation':True,
 'raw_receipt':raw.relative_to(ROOT).as_posix(),'raw_receipt_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
(base/'partial_control_results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
