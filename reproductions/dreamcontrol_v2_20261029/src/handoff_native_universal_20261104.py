"""Checkpoint the superseded autonomous adapter for the user's native-tracker switch.

The running PPO and every other project are left alone. No active source changes.
"""
import hashlib,json,os,signal,subprocess,time,fcntl,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103';W=D/'native_universal_20261104'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,j):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
def live(pid):
 try:return Path('/proc',str(pid),'stat').read_text().split()[2]!='Z'
 except FileNotFoundError:return False
def main():
 W.mkdir(exist_ok=True);assert not (W/'handoff_receipt.json').exists()
 state={'status':'checkpointing_superseded_autonomous_adapter','pid':os.getpid(),'PPO_unchanged':True}
 def save(**kw):state.update(kw);atomic(W/'status.json',state)
 active=read(D/'active_pick500_run.json');run=Path(active['run_dir']);s=read(run/'status.json')
 assert run.name=='pick500_8912_72000_parallel_v2' and s['status'] in ['online_training','failed']
 assert read(S/'runs/pick500_8912_3000_v1/status.json')['status']=='ppo_running'
 assert shutil.disk_usage('/data').free>=8*1024**3+356*1024**2+650*1024**2
 (W/'previous_active_pick500_run.json').open('x').write(json.dumps(active,indent=2))
 for pid,source in [(1448225,'stage3_supermimic_numerical_guard_v2.py'),(1448750,'stage3_supermimic_pick500_eval_followup_v2.py')]:
  if live(pid):
   assert source in Path('/proc',str(pid),'cmdline').read_bytes().decode();os.kill(pid,signal.SIGTERM)
 save(previous_run=str(run),previous_python_pid=s['pid'],request_at_vector_step=s['progress']['actual_vector_steps'])
 if s['status']=='online_training':
  (run/'REQUEST_STOP').open('x').write('User selected one native reference-conditioned universal tracker. Preserve this autonomous adapter as a prior result.\n')
 started=time.monotonic()
 while True:
  s=read(run/'status.json')
  if s['status'] in ['protected_stop','failed']:
   cp=s['latest_checkpoint'];assert cp['roundtrip'] and cp['vector_steps']>96 and sha(cp['path'])==cp['sha256'];break
  assert time.monotonic()-started<240,'preserve process; checkpoint request timed out'
  assert live(s['pid']),'process exited before protected checkpoint'
  time.sleep(2)
 group=os.getpgid(s['pid']) if live(s['pid']) else 1449772;assert group==1449772
 if live(s['pid']):os.killpg(group,signal.SIGTERM)
 for _ in range(30):
  if not live(s['pid']):break
  time.sleep(.5)
 if live(s['pid']):os.killpg(group,signal.SIGKILL)
 ppo=read(S/'runs/pick500_8912_3000_v1/status.json');assert ppo['status']=='ppo_running' and ppo['pid']==1436632
 receipt={'status':'superseded_adapter_checkpointed_for_native_universal_migration','user_choice':'one reference-conditioned universal tracker',
  'old_run':str(run),'old_run_terminal_status':s['status'],'old_run_error':s.get('error'),
  'saved_progress':s['progress'],'saved_counters':s['counters'],'checkpoint':cp,
  'PPO_pid_unchanged':ppo['pid'],'PPO_updates':ppo['policy_updates'],'no_task_success_claim':True,
  'native_policy_inputs':'proprioceptive history and future reference frames; explicitly permitted by latest user',
  'next':'reuse full upstream train.py/compile/batched updates on existing Pick500 references; hands/object task integration reported separately'}
 (W/'handoff_receipt.json').open('x').write(json.dumps(receipt,indent=2));save(**receipt)
 a=read(D/'adaptation_status.json');a.update(phase='native_universal_tracker_migration',formal_training_started=False,
  old_autonomous_adapter_checkpoint=cp,universal_reference_conditioned_user_authorized=True,native_work_directory=str(W));atomic(D/'adaptation_status.json',a)
 with (Q/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name in ['sequence.json','progress.json']:
   p=Q/name;j=read(p);j['SuperMimic_policy_mode']={'kind':'one_universal_reference_conditioned_tracker','initial_references':500,
    'planned_skills':['pick','pour','bimanual_box_lift','box_push_pull'],'planned_total_references':2000,
    'PPO_remains_task_specific':True,'old_autonomous_adapter_preserved':cp,'working_directory':str(W),
    'hands_object_interaction_not_yet_certified':True}
   if name=='progress.json':j['status']='Pick_PPO_running_native_universal_tracker_preparation'
   atomic(p,j)
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
