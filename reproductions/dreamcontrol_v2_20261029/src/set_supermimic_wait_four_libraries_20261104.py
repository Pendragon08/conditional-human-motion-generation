"""Apply the user's all-four-library dependency; preserve running Pick PPO."""
import datetime,fcntl,hashlib,json,pathlib,subprocess
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103';N=D/'native_universal_20261104';O=R/'stage3_pick29_20261103/reference_scale500_20261104'
def read(p):return json.loads(p.read_text())
def atomic(p,j):
 t=p.with_suffix('.waitfour.tmp');t.write_text(json.dumps(j,indent=2,ensure_ascii=False));t.replace(p)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();task_ids=['pick','pour','bimanual_box_lift','box_push_pull']
assert not read(D/'active_native_universal_run.json')['formal_training_started']
assert read(O/'final_library_audit.json')['status']=='passed'
manifest=O/'final_library_v1/reference_bank.json';mh=hashlib.sha256(manifest.read_bytes()).hexdigest();assert mh=='01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572'
ppo=read(R/'stage3_pick29_20261103/runs/pick500_8912_3000_v1/status.json');assert ppo['pid']==1436632 and ppo['status']=='ppo_running'
gate={'policy_version':1,'user_instruction_date':'2026-11-04','updated_utc':now,
 'status':'waiting_all_four_final_reference_libraries','training_allowed':False,'algorithm':'full_native_reference_conditioned_universal_SuperMimic',
 'required_tasks':task_ids,'required_per_task':500,'required_total':2000,'body_dof':29,'source_reference_fps':20,
 'requirements':['each task has500 distinct generated references with actual task-specific conditions and final admission evidence',
 'one versioned mixed manifest binds all four task manifests and their source/condition/state hashes',
 'actual mixed native loader/clock/joint mapping and resource budget verified'],
 'verified_counts':{'pick':500,'pour':0,'bimanual_box_lift':0,'box_push_pull':0},
 'pick_manifest':str(manifest),'pick_manifest_sha256':mh,
 'other_tasks_stage':'task-specific adapters and generated final references not yet complete',
 'native500_validation_only':True,'Pick_PPO_unchanged_pid':ppo['pid'],'Pick_PPO_updates_at_policy_change':ppo['policy_updates'],
 'next_action':'prioritize Pour, bimanual box lift, box push-pull reference generation; do not formally train native SuperMimic on Pick500 alone',
 'does_not_pause_Pick_PPO':True,'old_failures_and_validation_retained':True}
backup=N/'before_wait_all_four_20261104';backup.mkdir(exist_ok=False)
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for filename in ['sequence.json','progress.json']:
  p=Q/filename;(backup/filename).write_bytes(p.read_bytes());j=read(p)
  mode=j['SuperMimic_policy_mode'];mode.update(initial_references=2000,wait_for_all_four_reference_libraries=True,
   required_per_task=500,formal_gate=str(N/'all_four_reference_gate.json'),native500_validation_only=True)
  if filename=='sequence.json':
   j['version']=6;j['latest_user_instruction']='2026-11-04: Finish all four500-reference libraries, then train one universal native SuperMimic tracker on their2000-reference mixed library. Current Pick PPO continues.'
   j['policy'].update(wait_for_all2000_before_SuperMimic=True,wait_for_all2000_before_pick_ppo=False,
    preferred_backend_after_pick_evaluation='native universal reference tracker only after all four500 libraries are finalized')
   j['next_phase']='remaining_three_task_reference_generation'
   for t in j['tasks']:
    if t['id']=='pick':t.update(status='reference500_finalized_PPO_running',implementation='final500-reference library admitted; separate Pick PPO running',reference_bank=str(manifest),reference_bank_sha256=mh,run_dir=str(R/'stage3_pick29_20261103/runs/pick500_8912_3000_v1'),active_run_metadata=str(O/'active_pick500_run.json'))
   for comparison in j.get('comparisons',[]):
    if comparison['id']=='pick_supermimic':comparison.update(implementation='superseded_by_universal_all_four_native_tracker',preserved_as_history=True,prerequisite='all four500-reference libraries finalized; full mixed native input verified')
  else:
   j.update(status='Pick_PPO_running_SuperMimic_waiting_all_four_reference_libraries',next_action=gate['next_action'],next_phase='remaining_three_task_reference_generation')
   for t in j['tasks']:
    t['admitted_references']=gate['verified_counts'][t['id']]
    if t['id']=='pick':t.update(status='pick500_formal_PPO_running',new_PPO_started=True,reference_manifest=str(manifest),reference_manifest_sha256=mh)
  atomic(p,j)
atomic(N/'all_four_reference_gate.json',gate)
for p in [N/'status.json',D/'adaptation_status.json',D/'active_native_universal_run.json']:
 (backup/p.name).write_bytes(p.read_bytes());j=read(p);j.update(status=gate['status'],formal_training_started=False,
  wait_for_all_four_reference_libraries=True,formal_reference_gate=str(N/'all_four_reference_gate.json'))
 if p.name=='adaptation_status.json':j['phase']=gate['status']
 atomic(p,j)
(N/'wait_all_four_receipt.json').open('x').write(json.dumps(gate,indent=2,ensure_ascii=False));print(json.dumps(gate,indent=2,ensure_ascii=False))
