"""Preserve current worker output before a versioned initializer handoff."""
import hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
OUT=S/'box_initialization_handoff_v1'
PARENT=1916624
WORKER=1937140
EXPECTED=S/'new_references_paper_v8/box_push_pull/cohort_0000'
sys.path.insert(0,str(S/'src/paper_protocol_v8'))

def save(name,data):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

def cmd(pid):
 p=Path(f'/proc/{pid}/cmdline');return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''

def work():
 import pipeline_controller_v2 as p
 stopped=False
 try:
  assert 'paper_protocol_v8/pipeline_controller_v2.py' in cmd(PARENT) and '--postopt-worker' not in cmd(PARENT)
  assert str(EXPECTED/'postopt') in cmd(WORKER) and '--postopt-worker' in cmd(WORKER)
  save('before.json',{'parent':PARENT,'worker':WORKER,'parent_cmd':cmd(PARENT),'worker_cmd':cmd(WORKER),
                      'reason':'user requested existing-sample initialization verification before corrected production'})
  os.kill(PARENT,signal.SIGSTOP);stopped=True
  deadline=time.monotonic()+3600
  while cmd(WORKER):
   stat=Path(f'/proc/{WORKER}/stat').read_text().split()
   if stat[2]=='Z':break
   snap=p.resource_snapshot(S.parent,WORKER);p.resource_guard(snap)
   save('status.json',{'status':'waiting_current_postopt_to_finish','resources':snap,'worker':WORKER,'parent_paused':True})
   if time.monotonic()>deadline:raise TimeoutError('handoff phase timeout')
   time.sleep(2)
  rp=EXPECTED/'postopt/candidate_20300000/report.json'
  report=json.loads(rp.read_text())
  assert report['status'] in ('kinematic_candidate_ready_for_native_probe','generated_candidate_rejected')
  save('worker_complete.json',{'report':str(rp),'sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'status':report['status']})
 finally:
  if stopped and cmd(PARENT):
   os.kill(PARENT,signal.SIGINT);os.kill(PARENT,signal.SIGCONT)
  deadline=time.monotonic()+60
  while cmd(PARENT) and time.monotonic()<deadline:time.sleep(1)
  save('status.json',{'status':'prefix_preserved_for_versioned_handoff' if not cmd(PARENT) else 'parent_exit_pending',
                      'parent_live':bool(cmd(PARENT)),'worker_live':bool(cmd(WORKER)),
                      'worker_complete_receipt':str(OUT/'worker_complete.json')})

if __name__=='__main__':
 if '--worker' in sys.argv:
  try:work()
  except BaseException:
   (OUT/'traceback.txt').write_text(traceback.format_exc());raise
 else:
  if OUT.exists():print((OUT/'launch.json').read_text());raise SystemExit
  OUT.mkdir()
  with (OUT/'controller.log').open('xb') as f:
   child=subprocess.Popen([sys.executable,'-B','-u',__file__,'--worker'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  save('launch.json',{'pid':child.pid,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
  print((OUT/'launch.json').read_text())
