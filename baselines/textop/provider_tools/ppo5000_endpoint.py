"""Exclusive, evidence-bound CPU control of the user-selected PPO5000 endpoint."""
import argparse,ast,base64,hashlib,json,os,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
O=B/'results/step4_train_only_20261029/ppo5000_endpoint'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RUN='user_ppo5000_control_v1'
NAMES=['scripts/ppo5000_endpoint_protocol.py','scripts/watch_ppo5000_endpoint.py',
 'scripts/supervise_ppo5000_endpoint.py','scripts/test_ppo5000_endpoint.py']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fixture','enable','launch','collect'])
 p.add_argument('--role',choices=['author','document']);p.add_argument('--evidence',required=True);p.add_argument('--document-summary')
 a=p.parse_args();assert a.evidence.isascii() and a.evidence.isidentifier()
 O.mkdir(parents=True,exist_ok=True);out=O/a.evidence;out.mkdir(exist_ok=False)
 bundle={n:dict(sha256=sha(B/n),raw=base64.b64encode((B/n).read_bytes()).decode()) for n in NAMES}
 for n in NAMES:ast.parse((B/n).read_bytes(),feature_version=(3,8))
 summary=json.loads(pathlib.Path(a.document_summary).read_bytes()) if a.document_summary else None
 if a.action=='enable':
  assert summary['status']=='PPO5000_SECOND_FIXTURE_PARENT_REVIEW_VERIFIED' and summary['actual_exit_code']==0 and summary['findings']==[]
  assert summary['review_mode']=='parent_manual_source_review_plus_two_isolated_process_fixtures' and summary['independent_AI_agent_reviewed'] is False
  assert summary['watcher_sha256']==bundle['scripts/watch_ppo5000_endpoint.py']['sha256']
  assert summary['protocol_sha256']==bundle['scripts/ppo5000_endpoint_protocol.py']['sha256']
  assert summary['actual_production_signals']==0 and summary['actual_fixture_signals']==1
 code=r'''
import ast,base64,fcntl,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from step4_train_only_protocol import owned,read,sha,atomic
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from vae100k_control_protocol import identity,verify_identity
action=ACTION_LITERAL;role=ROLE_LITERAL;bundle=BUNDLE_LITERAL;summary=SUMMARY_LITERAL
run='user_ppo5000_control_v1';original='user_step4_train_only_ppo10000_train_v4'
parent=owned(r/'step34_numeric_v1');sp=r/'configs/ppo5000-user-control-spec.json'
req=r/'receipts'/('step4_train_only_'+original+'_request.json')
assert sha(req)=='715feef32953cc79491781753587aa0041dd154db12bfdaa8576d3f57de4aa82'
orig=read(req);assert orig['policy']['target_RL_iterations']==10000 and orig['policy']['training_authorized']
assert orig['executor_sha256']=='78d98f457e0a286263196ae523476895b71b56ae690a9c8ee889426805cf7bc8'
def actual_trainer():
 producer=identity(479092);sup=identity(478971)
 assert producer['start_ticks']=='1446080144' and producer['uid']==1013 and str(r/'scripts/run_step4_train_only_ppo_v4.py') in producer['argv'] and str(req) in producer['argv']
 assert sup['start_ticks']=='1446078760' and sup['uid']==1013 and str(r/'scripts/supervise_step4_train_only_ppo_v4.py') in sup['argv'] and str(req) in sup['argv']
 assert producer['state'] not in ['Z','X'] and sup['state'] not in ['Z','X']
 return producer,sup
def save_new(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
if action=='prepare':
 assert not sp.exists();producer,sup=actual_trainer()
 assert not (parent/run).exists() and not (parent/original/'emergency_last_good.private.pth').exists()
 for n,item in bundle.items():
  path=r/n;assert not path.exists();raw=base64.b64decode(item['raw']);assert sha_bytes(raw)==item['sha256'];ast.parse(raw,feature_version=(3,8))
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  assert sha(owned(path))==item['sha256']
 from ppo5000_endpoint_protocol import boundary
 rec=read(parent/original/'receipt.json');assert boundary(rec) is False
 bound={n:sha(r/n) for n in bundle}
 for n in ['scripts/vae100k_control_protocol.py','scripts/quick_pidfd.py','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py','scripts/run_step4_train_only_ppo_v4.py','scripts/supervise_step4_train_only_ppo_v4.py','scripts/step4_train_only_ppo_runtime_v4.py']:
  bound[n]=sha(owned(r/n))
 assert bound['scripts/run_step4_train_only_ppo_v4.py']==orig['executor_sha256']
 assert bound['scripts/supervise_step4_train_only_ppo_v4.py']==orig['supervisor_sha256']
 spec=dict(schema='textop.user_ppo5000_endpoint.v1',run=run,authorized=False,target_iterations=5000,target_optimizer_updates=100000,
  original_run=original,original_request_sha256=sha(req),producer=producer,producer_supervisor=sup,
  watcher_sha256=bound['scripts/watch_ppo5000_endpoint.py'],supervisor_sha256=bound['scripts/supervise_ppo5000_endpoint.py'],
  bound_sources=bound,author_fixture_binding=None,document_fixture_binding=None,document_summary_sha256=None,
  note='One pidfd SIGINT after persisted iteration5000 metadata; exact saved emergency state requires independent post-exit CPU audit')
 save_new(sp,spec);result=dict(status='PPO5000_CONTROL_PREPARED_DISABLED',spec=spec,spec_sha256=sha(sp),observed_iterations=rec['formal_completed_iterations'],production_signals=0)
else:
 assert sp.exists();s=read(sp)
 for n,item in bundle.items():assert sha(owned(r/n))==item['sha256']
 for n,d in s['bound_sources'].items():assert sha(owned(r/n))==d
 from ppo5000_endpoint_protocol import boundary
 if action=='fixture':
  assert not s['authorized'] and role in ['author','document']
  name='ppo5000_control_'+role+'_fixture_v1';folder=parent/name;log=parent/(name+'_supervision')
  assert not folder.exists() and not log.exists();log.mkdir(mode=0o700,exist_ok=False)
  cmd=['taskset','-c','0-3','/usr/bin/python3','-B',str(r/'scripts/test_ppo5000_endpoint.py'),'--out',str(folder)]
  proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',lambda pid:sample_resources(r,pid,1024**3,2*1024**3+128*1024**2),120,15)
  receipt=read(folder/'receipt.json') if (folder/'receipt.json').exists() else {}
  assert proof['supervision_passed'] and receipt['status']=='PPO5000_BOUNDARY_CPU_FIXTURE_VERIFIED'
  assert receipt['production_signals']==0 and receipt['fixture_signals']==1
  result=dict(status='PPO5000_CPU_FIXTURE_CLOSED',role=role,run=name,receipt=receipt,receipt_sha256=sha(folder/'receipt.json'),
   process_exit_sha256=sha(log/'process_exit.json'),resource_sha256=sha(log/'resources.jsonl'),actual_exit_code=proof['child_exit_code'],
   watcher_sha256=sha(r/'scripts/watch_ppo5000_endpoint.py'),protocol_sha256=sha(r/'scripts/ppo5000_endpoint_protocol.py'))
  save_new(folder/'accepted.json',result)
 elif action=='enable':
  assert not s['authorized'] and summary['status']=='PPO5000_SECOND_FIXTURE_PARENT_REVIEW_VERIFIED' and summary['findings']==[]
  assert summary['review_mode']=='parent_manual_source_review_plus_two_isolated_process_fixtures' and summary['independent_AI_agent_reviewed'] is False
  bindings={}
  for role in ['author','document']:
   name='ppo5000_control_'+role+'_fixture_v1';folder=parent/name;log=parent/(name+'_supervision');accepted=read(folder/'accepted.json');proof=read(log/'process_exit.json');receipt=read(folder/'receipt.json')
   assert proof['supervision_passed'] and proof['child_reaped'] and proof['child_exit_code']==0 and not proof['sample_errors']
   assert receipt['production_signals']==0 and receipt['fixture_signals']==1
   assert accepted['receipt_sha256']==sha(folder/'receipt.json') and accepted['process_exit_sha256']==sha(log/'process_exit.json')
   bindings[role]=dict(run=name,receipt_sha256=sha(folder/'receipt.json'),process_exit_sha256=sha(log/'process_exit.json'))
  assert summary['fixture_receipt_sha256']==bindings['document']['receipt_sha256']
  assert summary['watcher_sha256']==s['watcher_sha256'] and summary['protocol_sha256']==s['bound_sources']['scripts/ppo5000_endpoint_protocol.py']
  verify_identity(s['producer']);verify_identity(s['producer_supervisor']);assert boundary(read(parent/original/'receipt.json')) is False
  before=r/'receipts/ppo5000_control_spec_before_enable.json'
  with before.open('xb') as f:f.write(sp.read_bytes());f.flush();os.fsync(f.fileno())
  doc=r/'receipts/ppo5000_control_fresh_document_verified.json';save_new(doc,summary)
  s.update(authorized=True,author_fixture_binding=bindings['author'],document_fixture_binding=bindings['document'],document_summary_sha256=sha(doc))
  tmp=sp.with_name(sp.name+'.enable.partial')
  with tmp.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
  tmp.replace(sp);result=dict(status='PPO5000_CONTROL_ENABLED_NOT_DISPATCHED',spec_sha256=sha(sp),production_signals=0)
 elif action=='launch':
  assert s['authorized'] and s['author_fixture_binding'] and s['document_fixture_binding']
  verify_identity(s['producer']);verify_identity(s['producer_supervisor']);assert boundary(read(parent/original/'receipt.json')) is False
  assert not (parent/original/'emergency_last_good.private.pth').exists()
  prefix='ppo5000_control_'+run;paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_outer_exit.json']]
  assert not any(x.exists() for x in paths) and not (parent/run).exists()
  lock=(r/'receipts/user_ppo5000_control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  resources=sample_resources(r,None,1024**3,2*1024**3+128*1024**2)
  cmd=['taskset','-c','0-3','/usr/bin/python3','-B',str(r/'scripts/supervise_ppo5000_endpoint.py'),'--spec',str(sp)];outer=paths[2]
  text='import subprocess,pathlib,json,os\np=subprocess.run('+repr(cmd)+')\nwith pathlib.Path('+repr(str(outer))+').open("x") as f:json.dump({"actual_returncode":p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n'
  with paths[0].open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
  lock.close()
  with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['/usr/bin/python3','-B',str(paths[0])],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
  st=(pathlib.Path('/proc')/str(q.pid)/'stat').read_text().rsplit(')',1)[1].split()
  result=dict(status='PPO5000_CONTROL_DISPATCHED_NOT_YET_ARMED',run=run,driver_pid=q.pid,driver_start_ticks=st[19],spec_sha256=sha(sp),production_signals=0,resources=resources)
  save_new(paths[1],result)
 else:
  assert action=='collect';prefix='ppo5000_control_'+run;files={};hashes={}
  for name,path in dict(receipt=parent/run/'receipt.json',exit=r/'receipts'/(prefix+'_exit.json'),process=r/'receipts'/(prefix+'_process_exit.json'),outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
   if path.exists():files[name]=read(path);hashes[name]=sha(path)
  workers=[]
  for proc in pathlib.Path('/proc').iterdir():
   if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
   try:
    if proc.stat().st_uid!=1013:continue
    argv=(proc/'cmdline').read_bytes().decode().split('\0')
    if str(sp) in argv and any(str(r/'scripts'/n) in argv for n in ['watch_ppo5000_endpoint.py','supervise_ppo5000_endpoint.py']):workers.append(identity(int(proc.name)))
   except (FileNotFoundError,ProcessLookupError):pass
  result=dict(run=run,files=files,sha256=hashes,workers=workers,observed_unix=time.time())
print(json.dumps(result))
'''.replace('ROOT_LITERAL',repr(ROOT)).replace('ACTION_LITERAL',repr(a.action)).replace('ROLE_LITERAL',repr(a.role)).replace('BUNDLE_LITERAL',repr(bundle)).replace('SUMMARY_LITERAL',repr(summary))
 code=code.replace('def save_new(p,value):','def sha_bytes(raw):\n import hashlib\n return hashlib.sha256(raw).hexdigest()\ndef save_new(p,value):')
 ast.parse(code,feature_version=(3,8));(out/'remote_source.py').write_text(code,encoding='utf-8')
 result=json.loads(recorded_remote(code,out/'client.json',timeout=240))
 (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 if a.action=='collect':(O/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
