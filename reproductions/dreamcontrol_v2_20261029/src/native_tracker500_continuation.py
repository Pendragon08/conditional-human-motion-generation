"""One serial capacity+benchmark continuation after the native optimizer proof."""
import ast,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage3_supermimic29_20261103';W=D/'native_universal_20261104'
def read(p):return json.loads(Path(p).read_text())
state={'status':'waiting_full_native_smoke','pid':os.getpid(),'formal_training_started':False}
def save():
 t=W/'continuation.tmp';t.write_text(json.dumps(state,indent=2));t.replace(W/'continuation_status.json')
assert not (W/'continuation_status.json').exists();save()
try:
 start=time.monotonic()
 while True:
  s=read(W/'smoke_author_v1_guard.json')
  if s['status']=='failed_preserved':raise RuntimeError(s['error'])
  if s['status']=='passed':break
  assert time.monotonic()-start<2400;time.sleep(10)
 for mode in ['capacity','benchmark']:
  label=mode+'_8912_v1';source=D/'src/native_tracker500_capacity_guard.py';ast.parse(source.read_text())
  assert not (W/(label+'_guard.json')).exists()
  state.update(status=mode+'_running',phase=label);save();log=(W/(label+'_supervisor.log')).open('x')
  p=subprocess.Popen(['python3','-u',str(source),'--mode',mode,'--label',label],cwd=str(D),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  (W/(label+'_launch.json')).open('x').write(json.dumps({'pid':p.pid,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
  code=p.wait();log.close();s=read(W/(label+'_guard.json'));assert code==0 and s['status']=='passed'
  state[label+'_result']=s['worker_result'];save()
 state.update(status='native_capacity_and_compiled_benchmark_passed_formal_resource_evaluation_pending');save()
except BaseException as error:
 state.update(status='failed_preserved',error=repr(error),traceback=traceback.format_exc());save();raise
