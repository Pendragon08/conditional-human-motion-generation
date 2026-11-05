import ast,json,os,subprocess,fcntl,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103';W=D/'parallel_pick500_20261104';V=W/'numerical_repair_v2'
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
failed=D/'runs/pick500_8912_72000_parallel_v1/status.json';s=read(failed);assert s['status']=='failed'
receipt={'status':'failed_preserved','failure':s['error'],'failed_run':str(failed.parent),'failed_status_sha256':hashlib.sha256(failed.read_bytes()).hexdigest(),
 'last_confirmed_actual_vector_steps':s['progress']['actual_vector_steps'],'failure_step_not_persisted_in_v1':True,
 'unlogged_tail_upper_bound_steps':23,'optimization_checkpoint_before_failure':False,
 'PPO_unaffected_pid':1436632,'recovery':'isolated FP64 gradient-norm hypothesis and strict component diagnostics;600-step8912 test before one new formal version; no NaN masking'}
if not (W/'failure_v1_observed.json').exists():(W/'failure_v1_observed.json').open('x').write(json.dumps(receipt,indent=2))
for p in [D/'src/stage3_supermimic_pick500_eval_followup_v2.py',D/'src/stage3_supermimic_pick500_evaluate_v2.py']:ast.parse(p.read_text())
if not (V/'eval_followup_launch.json').exists():
 log=(V/'eval_followup.log').open('x');p=subprocess.Popen(['python3','-u',str(D/'src/stage3_supermimic_pick500_eval_followup_v2.py')],
  env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},cwd=str(D),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (V/'eval_followup_launch.json').open('x').write(json.dumps({'pid':p.pid,'status':str(V/'eval_followup_status.json')},indent=2))
a=read(D/'adaptation_status.json');a.update(phase='v1_gradient_failure_isolated_v2_stress_test',formal_training_started=False,
 previous_formal_failed=str(failed),repair_controller=str(V/'status.json'),repair_controller_pid=read(V/'status.json')['pid'],
 retry_is_new_run_not_resume=True,old_PPO_continues=True,proposed_v2_eval=str(V/'eval_followup_status.json'));atomic(D/'adaptation_status.json',a)
with (Q/'controller.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);q=read(Q/'progress.json');q.update(status='pick500_PPO_running_SuperMimic_numerical_repair',
  supermimic_failure=receipt,supermimic_repair_controller=str(V/'status.json'));atomic(Q/'progress.json',q)
print(json.dumps({'PPO':read(S/'runs/pick500_8912_3000_v1/status.json')['policy_updates'],'repair':read(V/'status.json'),
 'cpu_verification':read(V/'CPU_gradient_verification.json') if (V/'CPU_gradient_verification.json').exists() else None,
 'eval_followup':read(V/'eval_followup_launch.json')},indent=2))
