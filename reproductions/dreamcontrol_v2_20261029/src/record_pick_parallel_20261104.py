import ast,fcntl,hashlib,json,os,subprocess,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104';D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103';W=D/'parallel_pick500_20261104'
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
state=read(W/'controller_v1/status.json');assert state['pid']==1442525 and state['ppo_python_pid']==1436632
assert read(S/'runs/pick500_8912_3000_v1/status.json')['status']=='ppo_running'
evalsource=D/'src/stage3_supermimic_pick500_evaluate.py';follow=D/'src/stage3_supermimic_pick500_eval_followup.py'
ast.parse(evalsource.read_text());ast.parse(follow.read_text())
if not (W/'eval_followup_launch.json').exists():
 log=(W/'eval_followup.log').open('x')
 c=subprocess.Popen(['python3','-u',str(follow)],cwd=str(D),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt={'pid':c.pid,'source_sha256':hashlib.sha256(follow.read_bytes()).hexdigest(),'evaluation_source_sha256':hashlib.sha256(evalsource.read_bytes()).hexdigest(),'status':str(W/'eval_followup_status.json')}
 (W/'eval_followup_launch.json').open('x').write(json.dumps(receipt,indent=2))
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name in ['sequence.json','progress.json']:
  p=Q/name;j=read(p);backup=Q/(name+'.before_parallel_pick500_20261104')
  if not backup.exists():backup.open('x').write(json.dumps(j,indent=2))
  j['parallel_pick500_comparison']={'user_requested':True,'PPO_device':'cuda:1','SuperMimic_device':'cuda:0','references':500,
   'reference_manifest_sha256':'01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572',
   'controller':str(W/'controller_v1/status.json'),'controller_pid':1442525,'eval_followup':str(W/'eval_followup_status.json'),
   'PPO_updates':3000,'SuperMimic_real_vector_steps':72000,'SuperMimic_real_transitions':641664000,
   'same_assets_scene_reset_rewards_actor_actions':True,'per_job_GPU_GiB':12,'per_job_RSS_GiB':32,
   'total_GPU_GiB':24,'total_RSS_GiB':64,'no_duplicate_serial_SuperMimic_launch':True,'four_task_PPO_scope_unchanged':True}
  if name=='progress.json':j.update(controller_pid=1442525,status='pick500_PPO_running_parallel_SuperMimic_preparation',active_pick500_controller=str(W/'controller_v1/status.json'))
  atomic(p,j)
a=read(D/'adaptation_status.json');a.update(phase='same500_library_parallel_GPU0_controller',parallel_controller=str(W/'controller_v1/status.json'),
 formal_run_directory=str(D/'runs/pick500_8912_72000_parallel_v1'),eval_followup=str(W/'eval_followup_status.json'),
 same_reference_manifest_sha256='01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572',
 formal_training_started=state.get('supermimic_formal_started',False),no_PPO_interruption=True,previous_2reference_witnesses_not_repeated=True)
atomic(D/'adaptation_status.json',a)
print(json.dumps({'controller':read(W/'controller_v1/status.json'),'followup':read(W/'eval_followup_launch.json'),
 'PPO':read(S/'runs/pick500_8912_3000_v1/status.json'),'free_bytes':shutil.disk_usage('/data').free},indent=2))
