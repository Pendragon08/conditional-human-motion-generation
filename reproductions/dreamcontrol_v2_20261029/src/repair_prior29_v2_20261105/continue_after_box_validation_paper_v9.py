"""Wait for actual validation receipts, then deploy the verified continuation once."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
OUT=S/'box_validation_continuation_v9'
DEPLOY=S/'src/deploy_box_seed_resume_paper_v9.py'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def absent(pid):
 p=Path(f'/proc/{pid}/cmdline');return not p.exists() or not p.read_bytes()
def worker():
 try:
  launch=read(OUT/'launch.json')
  assert sha(DEPLOY)==launch['deploy_sha256']
  start=time.monotonic()
  while True:
   diag=S/'box_initialization_diagnostic_v1/status.json'
   hand=S/'box_initialization_handoff_v1/status.json'
   status=read(diag)['status'] if diag.exists() else 'pending'
   if status in ('failed_preserved','initialization_diagnostic_rejected','native_rejected'):
    save('status.json',{'status':'validation_not_passed_no_launch','diagnostic_status':status});return
   if status=='initialization_and_native_passed' and hand.exists() and read(hand)['status']=='prefix_preserved_for_versioned_handoff':
    if all(absent(pid) for pid in (1916624,1916625,1916626,1916627,1937140,1938522,1938529)):break
   save('status.json',{'status':'waiting_actual_validation_and_prefix','diagnostic_status':status,'new_generation_started':False})
   if time.monotonic()-start>10800:raise TimeoutError('three-hour bounded continuation wait')
   time.sleep(10)
  with (OUT/'deploy.log').open('xb') as f:
   result=subprocess.run([sys.executable,'-B','-u',str(DEPLOY),'--launch'],cwd=S,stdout=f,stderr=subprocess.STDOUT,timeout=900)
  if result.returncode:raise RuntimeError(f'deployer exit {result.returncode}; preserve deploy.log')
  final=S/'paper_pipeline_v9_launch/launch.json'
  assert read(final)['status']=='launched'
  save('status.json',{'status':'verified_continuation_launched','launch_receipt':str(final),'launch_sha256':sha(final)})
 except BaseException:
  (OUT/'traceback.txt').write_text(traceback.format_exc());save('status.json',{'status':'failed_preserved'});raise
if __name__=='__main__':
 if '--worker' in sys.argv:worker()
 else:
  if OUT.exists():print((OUT/'launch.json').read_text());raise SystemExit
  OUT.mkdir()
  with (OUT/'controller.log').open('xb') as f:
   proc=subprocess.Popen([sys.executable,'-B','-u',__file__,'--worker'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save('launch.json',{'pid':proc.pid,'source_sha256':sha(Path(__file__)),'deploy_sha256':sha(DEPLOY)})
  print((OUT/'launch.json').read_text())
