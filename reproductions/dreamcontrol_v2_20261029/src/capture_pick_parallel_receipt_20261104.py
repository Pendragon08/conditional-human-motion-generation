import json,hashlib,subprocess,shutil,time,fcntl
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';W=D/'parallel_pick500_20261104';Q=R/'stage3_skill_sequence_20261103'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,j):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
ppo_path=S/'runs/pick500_8912_3000_v1/status.json';sm_path=D/'runs/pick500_8912_72000_parallel_v1/status.json'
p=read(ppo_path);s=read(sm_path);c=read(W/'controller_v1/status.json')
compact={'PPO_updates':p['policy_updates'],'PPO_status':p['status'],'SM_status':s['status'],'SM_counters':s.get('counters'),
 'SM_progress':s.get('progress'),'controller_formal_started':c['supermimic_formal_started'],
 'PPO_resources':c['ppo_resources'],'SM_resources':c['parallel_child_resources'],'free_bytes':shutil.disk_usage('/data').free}
if s.get('counters',{}).get('policy_updates',0)>0 and s['status']=='online_training':
 assert s['formal_run'] and p['formal_training'] and p['pid']==1436632
 assert p['reference_manifest_sha256']==s['reference_manifest_sha256']=='01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572'
 assert s['config']['num_envs']==p['num_envs']==8912 and s['config']['macro_rounds']==p['max_iterations']==3000
 receipt={'status':'two_independent_Pick500_optimizers_running','unix_time':time.time(),'PPO':p,'SuperMimic':s,'controller':c,
  'same_manifest_asset_scene':True,'PPO_pid_not_restarted':True,'state_source_sha256':{'PPO':sha(ppo_path),'SM':sha(sm_path)},
  'source_binding':read(W/'controller_v1/source_binding.json'),'resource_snapshot':compact,
  'budget_vs_success':'actual optimization begun; no task success claim','scope':'simulation only'}
 if not (W/'parallel_training_started_receipt.json').exists():(W/'parallel_training_started_receipt.json').open('x').write(json.dumps(receipt,indent=2))
 a=read(D/'adaptation_status.json');a.update(formal_training_started=True,phase='Pick500_parallel_formal_BPTT_updates_observed',
  formal_training_receipt=str(W/'parallel_training_started_receipt.json'));atomic(D/'adaptation_status.json',a)
 with (Q/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);q=read(Q/'progress.json');q.update(status='pick500_PPO_and_SuperMimic_formal_running',
   parallel_training_receipt=str(W/'parallel_training_started_receipt.json'));atomic(Q/'progress.json',q)
 compact['started_receipt']=str(W/'parallel_training_started_receipt.json')
print(json.dumps(compact,indent=2))
