"""Isolated SDK signal-handler witness; never signal a production process.

This must run only under a resource-bounded supervisor holding the common
model/SDK lock. It initializes no model, training environment or optimizer.
"""
import argparse, ast, json, os, pathlib, signal, time, traceback
from isaaclab.app import AppLauncher
from step4_train_only_protocol import ROOT, owned, read, sha, atomic

p=argparse.ArgumentParser();p.add_argument('--request',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
sp=owned(a.request);s=read(sp)
assert os.getuid()==1013 and a.headless and a.device=='cuda:1'
assert s['authorized'] is True and s['role']=='isolated_sdk_signal_witness'
assert sha(__file__)==s['executor_sha256']
for name,digest in s['bound_sources'].items():assert sha(owned(ROOT/name))==digest
def handler_ast(path):
 node=ast.parse(path.read_text());return ast.dump(next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name=='interrupted'),include_attributes=False)
assert handler_ast(pathlib.Path(__file__))==handler_ast(owned(ROOT/'scripts/run_step4_train_only_ppo5000_natural_v2.py'))
parent=owned(s['output_root']);out=parent/s['run'];out.mkdir(mode=0o700,exist_ok=False)
rec=dict(status='STARTING_ISOLATED_SDK_SIGNAL_WITNESS',run=s['run'],output_writers_closed=False,
 source_sha256=sha(__file__),request_sha256=sha(sp),production_signals=0,self_signals=0,
 model_loads=0,optimizer_calls=0,formal_updates=0)
start=time.monotonic();app=None

def save():
 rec['elapsed_seconds']=time.monotonic()-start;atomic(out/'receipt.json',rec)

def interrupted(signum,frame):
 raise RuntimeError('Owned Stage4 PPO interrupted with signal '+str(signum))

save()
try:
 signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
 app=AppLauncher(a).app
 rec['sdk_replaced_SIGINT']=signal.getsignal(signal.SIGINT) is not interrupted
 rec['sdk_replaced_SIGTERM']=signal.getsignal(signal.SIGTERM) is not interrupted
 signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
 assert signal.getsignal(signal.SIGINT) is interrupted and signal.getsignal(signal.SIGTERM) is interrupted
 assert app.is_running()
 actual=[]
 for sig in (signal.SIGINT,signal.SIGTERM):
  try:os.kill(os.getpid(),sig)
  except RuntimeError as e:
   assert str(e)=='Owned Stage4 PPO interrupted with signal '+str(int(sig))
   actual.append(int(sig));rec['self_signals']+=1
  else:raise AssertionError('Expected self-signal handler did not raise')
  assert app.is_running() and signal.getsignal(sig) is interrupted
 rec.update(status='ISOLATED_SDK_SIGNAL_WITNESS_FINISHED',delivered_self_signals=actual,
  running_after_each_signal=True,handler_identity_verified=True,output_writers_closed=True)
 save()
except BaseException as e:
 rec.update(status='ISOLATED_SDK_SIGNAL_WITNESS_FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
finally:
 if app is not None:
  from bounded_headless_close import close
  close(app,rec,save,out,simulation=None,successful_statuses=('ISOLATED_SDK_SIGNAL_WITNESS_FINISHED',))
