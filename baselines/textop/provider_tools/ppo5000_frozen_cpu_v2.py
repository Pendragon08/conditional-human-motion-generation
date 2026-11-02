"""Deploy disabled CPU source, then bind/launch/collect only a real PPO5000 stop."""
import argparse,ast,base64,hashlib,json,pathlib,shutil
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
O=B/'results/step4_train_only_20261029/ppo5000_frozen_cpu'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
SPEC=B/'configs/step4-ppo5000-frozen-cpu-v3-spec.json'
NEW=['scripts/step4_ppo5000_frozen_protocol_v2.py','scripts/step4_ppo5000_cpu_core_v2.py',
 'scripts/run_ppo5000_frozen_cpu_v2.py','scripts/supervise_ppo5000_frozen_cpu_v2.py']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','activate','launch','collect'])
 p.add_argument('--stage',choices=['export','verify']);p.add_argument('--evidence',required=True);a=p.parse_args()
 assert a.evidence.isascii() and a.evidence.isidentifier()
 assert (a.action=='prepare' and a.stage is None) or (a.action!='prepare' and a.stage in ['export','verify'])
 assert shutil.disk_usage(B).free>=2*1024**3+16*1024**2,'C local evidence reserve unavailable'
 raw=SPEC.read_bytes();spec=json.loads(raw)
 assert spec['authorized'] is False and spec['training_binding'] is spec['export_binding'] is None
 assert spec['expected_completed_iterations']==5000 and spec['expected_finite_Adam_updates']==100000
 for n,digest in spec['bound_sources'].items():
  assert sha(B/n)==digest
  if n.endswith('.py'):ast.parse((B/n).read_bytes(),feature_version=(3,8))
 bundle={n:dict(raw=base64.b64encode((B/n).read_bytes()).decode(),sha256=sha(B/n)) for n in NEW}
 O.mkdir(parents=True,exist_ok=True);out=O/a.evidence;out.mkdir(exist_ok=False)
 code=r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from step4_train_only_protocol import owned,read,sha
from contact_filter_resources import sample_resources
action=ACTION_LITERAL;stage=STAGE_LITERAL;bundle=BUNDLE_LITERAL;spec=SPEC_LITERAL
template=r/'configs/step4-ppo5000-frozen-cpu-v3-spec.json';parent=owned(r/'step34_numeric_v1')
expected_template=SPEC_SHA_LITERAL
assert sha(r/'receipts/step4_train_only_user_step4_train_only_ppo10000_train_v4_request.json')=='715feef32953cc79491781753587aa0041dd154db12bfdaa8576d3f57de4aa82'
def save_new(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 fd=os.open(str(path.parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
if action=='prepare':
 assert not template.exists()
 assert shutil.disk_usage(parent).free>=2*1024**3+128*1024**2+2*1024**2
 for n,item in bundle.items():
  path=r/n;assert not path.exists();raw=base64.b64decode(item['raw']);assert hashlib.sha256(raw).hexdigest()==item['sha256'];ast.parse(raw,feature_version=(3,8))
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  assert sha(owned(path))==item['sha256']
 for n,digest in spec['bound_sources'].items():assert sha(owned(r/n))==digest
 raw=base64.b64decode(TEMPLATE_LITERAL);assert hashlib.sha256(raw).hexdigest()==expected_template
 with template.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert sha(template)==expected_template and read(template)['authorized'] is False
 result=dict(status='PPO5000_FROZEN_CPU_DISABLED_SOURCES_DEPLOYED',template_sha256=sha(template),
  source_sha256={n:sha(r/n) for n in bundle},new_model_loads=0,new_optimizer_updates=0,signals=0)
 save_new(r/'receipts/ppo5000_frozen_cpu_preparation_v3.json',result)
else:
 assert sha(owned(template))==expected_template and read(template)==spec and spec['authorized'] is False
 for n,digest in spec['bound_sources'].items():assert sha(owned(r/n))==digest
 from step4_ppo5000_frozen_protocol_v2 import accepted_stop,accepted_export,fingerprint
 run='user_step4_ppo5000_frozen_'+stage+'_v1';prefix='ppo5000_frozen_'+run
 request=r/'receipts'/(prefix+'_request.json');log=parent/(run+'_supervision');out=parent/run
 lock=(r/'receipts/full_library_capacity.lock').open('a+')
 if action in ['activate','launch']:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  train=accepted_stop()
  export=accepted_export() if stage=='verify' else None
  required=2*1024**3+128*1024**2+(32*1024**2 if stage=='export' else 0)
  assert shutil.disk_usage(parent).free>=required
  resources=sample_resources(r,None,16*1024**3,required)
  assert not out.exists() and not log.exists()
  if action=='activate':
   assert not request.exists() and not any((r/'receipts'/(prefix+x)).exists() for x in ['_driver.py','_driver.json','_outer_exit.json'])
   job=dict(schema='textop.ppo5000_frozen_cpu_job.v1',authorized=True,role=stage,run=run,
    training_binding=train['binding'],export_binding=export['binding'] if export else None,
    executor_sha256=spec['executor_sha256'],supervisor_sha256=spec['supervisor_sha256'],
    bound_sources=spec['bound_sources'],source_template_sha256=expected_template,
    user_target_iterations=5000,user_target_optimizer_updates=100000,
    checkpoint_steps_pending_real_CPU_audit=True)
   save_new(request,job);result=dict(status='PPO5000_FROZEN_CPU_REQUEST_ACTIVATED_NOT_LAUNCHED',role=stage,
    run=run,request_sha256=sha(request),training_binding=train['binding'],new_model_loads=0,resources=resources)
  else:
   assert action=='launch' and request.exists();job=read(request)
   assert job['authorized'] is True and job['role']==stage and job['training_binding']==train['binding']
   assert job['bound_sources']==spec['bound_sources'] and job['source_template_sha256']==expected_template
   if stage=='verify':assert job['export_binding']==export['binding']
   paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_outer_exit.json']]
   assert not any(x.exists() for x in paths),'Inspect existing intent/outcome, never redispatch'
   cmd=['/usr/bin/python3','-B',str(r/'scripts/supervise_ppo5000_frozen_cpu_v2.py'),'--request',str(request)]
   text='import subprocess,pathlib,json,os\np=subprocess.run('+repr(cmd)+')\nwith pathlib.Path('+repr(str(paths[2]))+').open("x") as f:json.dump({"actual_returncode":p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n'
   with paths[0].open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
   lock.close()
   with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['/usr/bin/python3','-B',str(paths[0])],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
   st=(pathlib.Path('/proc')/str(q.pid)/'stat').read_text().rsplit(')',1)[1].split()
   result=dict(status='PPO5000_FROZEN_CPU_DISPATCHED_NOT_ACCEPTED',run=run,role=stage,driver_pid=q.pid,
    driver_start_ticks=st[19],request_sha256=sha(request),resources=resources)
   save_new(paths[1],result)
 else:
  assert action=='collect' and request.exists();job=read(request);files={};hashes={}
  for key,path in dict(receipt=out/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
   if path.exists():files[key]=read(path);hashes[key]=sha(path)
  workers=[]
  for proc in pathlib.Path('/proc').iterdir():
   if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
   try:
    if proc.stat().st_uid!=1013:continue
    argv=(proc/'cmdline').read_bytes().decode().split('\0')
    if str(request) in argv and any(str(r/'scripts'/n) in argv for n in ['run_ppo5000_frozen_cpu_v2.py','supervise_ppo5000_frozen_cpu_v2.py']):
     st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
   except (FileNotFoundError,ProcessLookupError):pass
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
  except BlockingIOError:free=False
  closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_reaped'] and files['process']['child_exit_code']==0 and not workers and free
  audit=None
  if closed:
   rec=files['receipt'];expected='PPO5000_FROZEN_CPU_EXPORTED' if stage=='export' else 'PPO5000_FROZEN_FRESH_CPU_VERIFIED'
   assert rec['status']==expected and rec['output_writers_closed'] and rec['fullstate_exact_5000_100000']
   assert rec['independent_actor_RMS_reload_exact'] and rec['full_model_and_cpu_Adam_restored_exact'] and rec['synthetic_probe_only']
   assert rec['optimizer_updates_performed']==rec['new_CUDA_contexts']==0 and rec['CUDA_RNG_restored'] is False
   assert rec['heldout_tracking_accepted'] is False and rec['online_latency_accepted'] is False
   if stage=='export':
    accepted=accepted_export();audit=dict(export_binding=accepted['binding'],real_5000_fullstate_cpu_audited=True,heldout_quality_pending=True)
   else:
    prior=accepted_export(job['export_binding']);ex=files['exit'];proc=files['process']
    assert ex['request_sha256']==sha(request) and ex['receipt_sha256']==hashes['receipt'] and ex['process_exit_sha256']==hashes['process']
    assert ex['resource_sha256']==sha(log/'resources.jsonl') and ex['source_sha256']==job['supervisor_sha256']
    assert proc['sample_errors']==[] and not proc['timed_out'] and proc['shutdown_actions']==[]
    rows=(log/'resources.jsonl').read_text().splitlines();last=json.loads(rows[-1]);assert last['phase']=='after_exit' and last['owned_group_processes']==last['owned_group_gpu_memory']==[]
    assert rec['training_binding']==prior['receipt']['training_binding'] and rec['inference']==prior['receipt']['inference']
    assert rec['audit']['fullstate_content_sha256']==prior['receipt']['audit']['fullstate_content_sha256']
    assert rec['audit']['actor_RMS_probe_output_hashes']==prior['receipt']['audit']['actor_RMS_probe_output_hashes']
    audit=dict(fresh_cpu_actor_RMS_exact=True,fullstate_exact_5000_100000=True,heldout_quality_pending=True)
  result=dict(run=run,role=stage,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,
   common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit)
 lock.close()
print(json.dumps(result))
'''.replace('ROOT_LITERAL',repr(ROOT)).replace('ACTION_LITERAL',repr(a.action)).replace('STAGE_LITERAL',repr(a.stage)).replace('BUNDLE_LITERAL',repr(bundle)).replace('SPEC_LITERAL',repr(spec)).replace('SPEC_SHA_LITERAL',repr(sha(SPEC))).replace('TEMPLATE_LITERAL',repr(base64.b64encode(raw).decode()))
 ast.parse(code,feature_version=(3,8));(out/'remote_source.py').write_text(code,encoding='utf-8')
 result=json.loads(recorded_remote(code,out/'client.json',timeout=240))
 (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 if a.action=='collect':(O/(a.stage+'_progress.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result))
if __name__=='__main__':main()
