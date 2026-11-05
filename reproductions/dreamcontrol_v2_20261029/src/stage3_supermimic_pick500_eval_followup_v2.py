"""Evaluate the completed parallel comparison, preserving independent PPO results."""
import hashlib,json,os,subprocess,signal,time,shutil,sys,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103'
W=D/'parallel_pick500_20261104/numerical_repair_v2';OUT=W/'evaluation100_v2';RUN=D/'runs/pick500_8912_72000_parallel_v2';MAN=S/'reference_scale500_20261104/final_library_v1/reference_bank.json'
sys.path.insert(0,str(D/'src'))
from stage3_pick_parallel_controller import resources,alive,stop_group
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(j):
 t=W/'eval_followup_status.tmp';t.write_text(json.dumps(j,indent=2));t.replace(W/'eval_followup_status.json')
source=D/'src/stage3_supermimic_pick500_evaluate_v2.py';source_sha=sha(source)
state={'pid':os.getpid(),'status':'waiting_completed_SuperMimic_budget','source_sha256':source_sha};save(state)
began=time.monotonic();child=None
try:
 while True:
  controller=read(W/'status.json')
  if controller.get('status')=='failed_preserved':
   state.update(status='training_failure_preserved_no_evaluation',training_error=controller.get('error'));save(state);sys.exit(1)
  if (RUN/'status.json').exists():
   s=read(RUN/'status.json')
   if s['status']=='failed':raise RuntimeError('SuperMimic training failed')
   if s['status']=='interaction_budget_completed' and s['progress']['actual_vector_steps']==72000:
    pids=read(W/'formal_v2_launch.json')['pid']
    if not alive(pids):break
  assert time.monotonic()-began<80*3600
  time.sleep(20)
 s=read(RUN/'status.json');checkpoint=Path(s['latest_checkpoint']['path'])
 assert s['formal_run'] and s['reference_count']==500 and s['latest_checkpoint']['roundtrip']
 assert sha(checkpoint)==s['latest_checkpoint']['sha256'] and sha(MAN)==s['reference_manifest_sha256']
 assert sha(source)==source_sha and not OUT.exists()
 spec=read(D/'runtime_bridge_spec.json');log=(W/'evaluation100_v2.log').open('x')
 command=[spec['python'],'-u',str(source),'--headless','--device','cuda:0','--checkpoint',str(checkpoint),'--training-run',str(RUN),
  '--reference-manifest',str(MAN),'--output-dir',str(OUT),'--num-envs','100','--steps','500']
 child=subprocess.Popen(command,cwd=spec['cwd'],env={**os.environ,**spec['env']},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (W/'evaluation100_v2_launch.json').open('x').write(json.dumps({'pid':child.pid,'command':command,'checkpoint_sha256':sha(checkpoint)},indent=2))
 state.update(status='read_only_100env_evaluation',child_pid=child.pid);save(state);started=time.monotonic();finished=None
 while child.poll() is None:
  own=resources(child.pid);assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3
  assert shutil.disk_usage('/data').free>=8*1024**3+356*1024**2
  if (OUT/'failure.json').exists():raise RuntimeError(read(OUT/'failure.json'))
  if (OUT/'report.json').exists():
   r=read(OUT/'report.json');assert r['status']=='evaluation_completed'
   if finished is None:finished=time.monotonic()
   if time.monotonic()-finished>40:stop_group(child.pid);break
  assert time.monotonic()-started<1800
  time.sleep(5)
 r=read(OUT/'report.json');assert r['num_envs']==100 and r['reference_count']==500 and r['policy_state_hash_unchanged']
 assert r['checkpoint_sha256']==sha(checkpoint) and r['policy_updates_during_evaluation']==0
 completion={'status':'interaction_budget_completed_evaluation_archived','algorithm':'SuperMimic-style adaptation',
  'task':'pick','references':500,'real_vector_steps':72000,'real_transitions':641664000,
  'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'reference_manifest_sha256':sha(MAN),
  'evaluation':str(OUT/'report.json'),'evaluation_sha256':sha(OUT/'report.json'),
  'success_count':r['success_count'],'success_ratio':r['success_ratio'],'success_definition':r['success_definition'],
  'budget_completion_is_not_task_success':True,'scope':'simulation only; independent comparison with PPO'}
 target=Q/'comparisons';target.mkdir(exist_ok=True)
 (target/'pick500_supermimic_parallel_v2.json').open('x').write(json.dumps(completion,indent=2))
 state.update(status='evaluation_archived',completion=completion);save(state)
except BaseException as error:
 if child and child.poll() is None:stop_group(child.pid)
 if not isinstance(error,SystemExit):state.update(status='failed_preserved',error=repr(error),traceback=traceback.format_exc());save(state)
 raise
