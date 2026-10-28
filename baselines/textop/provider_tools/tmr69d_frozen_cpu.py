"""Activate, uniquely dispatch, or collect future trained evaluator CPU jobs.

Activation refuses any running/incomplete training. All SSH calls retain their
own intent, direct streams and outcome; an unknown dispatch is never repeated.
"""
import argparse
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B/'results/semantic_evaluator_20261029'
p = argparse.ArgumentParser()
p.add_argument('action', choices=['activate', 'launch', 'collect'])
p.add_argument('--stage', choices=['export', 'verify'], required=True)
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
sp = B/('configs/tmr69d-frozen-'+a.stage+'-spec.json')
raw = sp.read_bytes()
spec = json.loads(raw)
assert spec['role'] == a.stage and spec['run'] == 'tmr69d_frozen_'+a.stage+'_v1'
out = O/(spec['run'] if a.action != 'activate' else 'tmr_frozen_'+a.stage+'_activation_v1')
if a.action == 'launch':
    assert spec['authorized'] is True
    out.mkdir(exist_ok=False)
else:
    out.mkdir(exist_ok=True)
sources = {name:hashlib.sha256((B/name).read_bytes()).hexdigest() for name in spec['bound_sources']}
assert sources == spec['bound_sources']
assert hashlib.sha256((B/'scripts/run_tmr69d_frozen_cpu.py').read_bytes()).hexdigest() == spec['executor_sha256']
assert hashlib.sha256((B/'scripts/supervise_tmr69d_frozen_cpu.py').read_bytes()).hexdigest() == spec['supervisor_sha256']
expected = hashlib.sha256(raw).hexdigest()
export_binding = None
if a.action == 'activate':
    assert spec['authorized'] is False and spec['training_binding'] is None
    assert spec['export_binding'] is None
    if a.stage == 'verify':
        previous = json.loads((O/'tmr69d_frozen_export_v1/progress.json').read_bytes())
        assert previous['closed_execution_verified'] and previous['artifact_audit']['export_binding']
        export_binding = previous['artifact_audit']['export_binding']

code = r'''
import ast,base64,fcntl,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));os.sched_setaffinity(0,{0,1,2,3})
from tmr69d_trained_acceptance import (ROOT,HOME,PARENT,owned,read,sha,accepted_training,accepted_frozen_export,closed_evaluator_job,fingerprint)
from contact_filter_resources import sample_resources
action=ACTION_LITERAL;stage=STAGE_LITERAL;sp=owned(r/('configs/tmr69d-frozen-'+stage+'-spec.json'))
assert sha(sp)==EXPECTED_LITERAL;s=read(sp);run=s['run'];prefix='tmr_frozen_cpu_'+run
for name,value in s['bound_sources'].items():assert sha(owned(r/name))==value,name
assert sha(owned(r/'scripts/run_tmr69d_frozen_cpu.py'))==s['executor_sha256']
assert sha(owned(r/'scripts/supervise_tmr69d_frozen_cpu.py'))==s['supervisor_sha256']
parent=owned(PARENT);log=parent/(run+'_supervision')
lock=(r/'receipts/full_library_capacity.lock').open('a+')
if action in ['activate','launch']:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 train=accepted_training(s['training_binding'])
 item=pathlib.Path(s['inference_path']);home=owned(item.parent if item.parent.exists() else item.parent.parent,HOME)
 assert subprocess.check_output(['findmnt','-T',str(home),'-n','-o','FSTYPE'],text=True).strip()=='ext4'
 home_required=s['reserve_bytes']+s['home_log_allowance_bytes']+(s['inference_upper_bytes'] if stage=='export' else 0)
 assert shutil.disk_usage(home).free>=home_required
 resources=sample_resources(r,None,16*1024**3,2*1024**3+128*1024**2+16*1024**2)
 assert not (parent/run).exists() and not log.exists()
 if action=='activate':
  assert s['authorized'] is False and s['training_binding'] is None and s['export_binding'] is None
  if stage=='verify':accepted_frozen_export(EXPORT_BINDING_LITERAL)
  else:assert not item.parent.exists();item.parent.mkdir(mode=0o700);owned(item.parent,HOME)
  archive=r/'receipts'/('tmr_frozen_'+stage+'_spec_before_enable.json')
  with archive.open('xb') as f:f.write(sp.read_bytes());f.flush();os.fsync(f.fileno())
  s.update(authorized=True,training_binding=train['binding'],export_binding=EXPORT_BINDING_LITERAL)
  raw=(json.dumps(s,indent=2)+'\n').encode();temp=sp.with_name(sp.name+'.enable.partial')
  with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  temp.replace(sp)
  result=dict(status='TMR_FROZEN_CPU_ENABLED_NOT_DISPATCHED',stage=stage,run=run,spec_sha256=sha(sp),
   spec_base64=base64.b64encode(raw).decode(),training_binding=train['binding'],new_model_loads=0,
   capacity=dict(home_free_bytes=shutil.disk_usage(home).free,home_required_bytes=home_required,data=resources))
  with (r/'receipts'/('tmr_frozen_'+stage+'_activation.json')).open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
 elif action=='launch':
  assert s['authorized'] is True
  if stage=='verify':accepted_frozen_export(s['export_binding'])
  paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_outer_exit.json']]
  assert not any(x.exists() for x in paths),'Inspect existing intent/outcome before redispatch'
  cmd=['python3',str(r/'scripts/supervise_tmr69d_frozen_cpu.py'),'--request',str(sp)]
  outer=r/'receipts'/(prefix+'_outer_exit.json');driver=paths[0]
  text="import subprocess,json,pathlib,os\np=subprocess.run("+repr(cmd)+")\nwith pathlib.Path("+repr(str(outer))+").open('x') as f:json.dump({'actual_returncode':p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n"
  with driver.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
  lock.close()
  with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['python3',str(driver)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
  result=dict(status='TMR_FROZEN_CPU_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,spec_sha256=sha(sp),resources=resources)
  with paths[1].open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
else:
 assert action=='collect' and s['authorized'] is True
 files={};hashes={}
 for key,path in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
  if path.exists():files[key]=read(path);hashes[key]=sha(path)
 workers=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   if proc.stat().st_uid!=1013:continue
   argv=(proc/'cmdline').read_bytes().decode().split('\0')
   if str(sp) in argv and any(str(r/'scripts'/n) in argv for n in ['run_tmr69d_frozen_cpu.py','supervise_tmr69d_frozen_cpu.py']):
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=stat[19],state=stat[0],argv=argv))
  except (FileNotFoundError,ProcessLookupError):pass
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
 except BlockingIOError:free=False
 closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_reaped'] and files['process']['child_exit_code']==0 and not workers and free
 audit=None
 if closed:
  rec=files['receipt'];expected='TMR69D_TRAINED_FROZEN_CPU_EXPORTED' if stage=='export' else 'TMR69D_TRAINED_FROZEN_FRESH_CPU_VERIFIED'
  assert rec['status']==expected and rec['output_writers_closed'] is True
  closure=closed_evaluator_job(run,parent/run/'receipt.json',hashes['receipt'],s['supervisor_sha256'],sha(sp))
  train=accepted_training(s['training_binding']);assert rec['training_binding']==train['binding']
  assert rec['source_sha256']==s['executor_sha256'] and rec['spec_sha256']==sha(sp)
  for name in ['full_model_and_cpu_adamw_restored_exact','independent_inference_reload_exact','synthetic_mean_forward_exact','original_fullstate_unchanged']:assert rec[name] is True
  assert rec['training_optimizer_updates_performed']==rec['new_cuda_contexts']==0 and rec['cuda_rng_restored'] is False and rec['semantic_metrics_accepted'] is False
  fp=fingerprint(rec['inference']['path'],home=True);assert {k:fp[k] for k in ['path','bytes','sha256']}==rec['inference']
  if stage=='export':
   binding=dict(run=run,spec_sha256=sha(sp),receipt_sha256=hashes['receipt'],closure=closure,inference=rec['inference'])
   accepted_frozen_export(binding)
   audit=dict(export_binding=binding,trained_fullstate_and_cpu_inference_verified=True,model_loads_by_collector=0,semantic_metrics_pending=True)
  else:
   author=accepted_frozen_export(s['export_binding'])
   for name in ['training_binding','inference','trained_model_sha256','fullstate_metadata_audit','synthetic_probe']:
    assert rec[name]==author['receipt'][name],name
   audit=dict(fresh_cpu_fullstate_and_inference_exact=True,closure=closure,export_binding=s['export_binding'],model_loads_by_collector=0,semantic_metrics_pending=True)
 logs={}
 for path in [log/'child.log',parent/(run+'_outer.log')]:
  if path.exists():
   with path.open('rb') as f:f.seek(max(0,path.stat().st_size-3000));logs[path.name]=f.read().decode(errors='replace')
 result=dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs)
lock.close();print(json.dumps(result))
'''.replace('ACTION_LITERAL',repr(a.action)).replace('STAGE_LITERAL',repr(a.stage)).replace('EXPECTED_LITERAL',repr(expected)).replace('EXPORT_BINDING_LITERAL',repr(export_binding))
ast.parse(code, feature_version=(3,8))
j = json.loads(recorded_remote(code, out/(a.evidence+'_client.json'), timeout=180))
(out/(a.evidence+'.json')).write_text(json.dumps(j,indent=2),encoding='utf-8')
if a.action == 'activate':
    updated = base64.b64decode(j['spec_base64'])
    assert hashlib.sha256(updated).hexdigest() == j['spec_sha256']
    with (out/'spec_before_enable.json').open('xb') as stream:
        stream.write(raw)
    sp.write_bytes(updated)
elif a.action == 'collect':
    (out/'progress.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
else:
    (out/'dispatch.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))
