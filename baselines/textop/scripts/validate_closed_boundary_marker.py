"""One CPU-only owned fixture reproduces the frozen trainer's exception pattern."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time
from closed_boundary_marker import request_boundary_save
from closed_boundary_interrupt import identity
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
assert a.run.isascii() and a.run.isidentifier() and os.getuid()==1013
out=ROOT/'pause_marker_fixtures'/a.run;out.mkdir(parents=True,exist_ok=False)
work=out/'fixture_run';work.mkdir();evidence=out/'evidence';evidence.mkdir()
src=out/'fixture.py';src.write_text('''import pathlib,json,time,sys,os
out=pathlib.Path(sys.argv[1]);boundary=False;step=0
def save():
 p=out/'receipt.partial';p.write_text(json.dumps(dict(status='RUNNING',step=step)));p.replace(out/'receipt.json')
save()
try:
 while step<100000:
  boundary=False;step+=4;time.sleep(.002);boundary=True
  if step%20==0:save()
except BaseException:
 if boundary:
  with (out/'simulated_emergency.json').open('x') as f:
   json.dump(dict(step=step,boundary=True,simulation_only=True),f);f.flush();os.fsync(f.fileno())
 save()
 raise
finally:(out/'finally_closed').write_text('yes')
''',encoding='utf-8')
with (out/'fixture.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,str(src),str(work)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while not(work/'receipt.json').exists():assert child.poll() is None;time.sleep(.01)
 expected=identity(child.pid)
 result=request_boundary_save(expected,work,evidence,30);rc=child.wait(timeout=5)
saved=json.loads((work/'simulated_emergency.json').read_text());text=(out/'fixture.log').read_text()
assert rc==1 and saved['boundary'] and saved['step']%20==0 and 'IsADirectoryError' in text
assert (work/'finally_closed').read_text()=='yes' and not(work/'receipt.partial').exists()
assert json.loads((work/'receipt.json').read_text())['step']<saved['step']
assert result['training_signals']==0 and result['source_edits']==0 and result['marker_removed']
r={'status':'CPU_CLOSED_BOUNDARY_MARKER_PASSED','actual_child_exit':rc,'outer_expected_exit':0,
 'checks':{'closed_full_batch':True,'real_emergency_file_before_final_io_failure':True,'original_receipt_retained':True,
 'finally_closed':True,'owned_empty_marker_removed_after_exit':True,'no_training_or_fixture_signals':True},
 'fixture_saved':saved,'control':result,'neural_checkpoint_created':False,'gpu_calls':0,
 'source_sha256':{n:hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest() for n in ['closed_boundary_marker.py','validate_closed_boundary_marker.py']}}
(out/'receipt.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))
