"""Prepare, fixture, enable, uniquely arm and lightly collect user100-epoch stop."""
import argparse,ast,base64,hashlib,json,os,pathlib
from backup_filter_recovery_checkpoint import recorded_remote
B=pathlib.Path(__file__).resolve().parents[1]
O=B/'results/semantic_evaluator_20261029'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RUN='user_tmr100_control_v1'
NAMES=['scripts/tmr100_endpoint_protocol.py','scripts/watch_tmr100_endpoint.py',
    'scripts/supervise_tmr100_endpoint.py','scripts/test_tmr100_endpoint.py']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def persist(p,data):
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fixture','inspect_fixture','enable','launch','collect'])
    p.add_argument('--role',choices=['author','document']);p.add_argument('--evidence',required=True)
    p.add_argument('--document-summary');a=p.parse_args()
    assert a.evidence.isascii() and a.evidence.isidentifier()
    out=O/('tmr100_endpoint_preparation_v1' if a.action in ['prepare','fixture','inspect_fixture','enable'] else RUN)
    out.mkdir(parents=True,exist_ok=True)
    spec_path=B/'configs/tmr100-user-control-spec.json'
    spec=json.loads(spec_path.read_bytes()) if spec_path.exists() else None
    sources={n:dict(sha256=sha(B/n),raw=base64.b64encode((B/n).read_bytes()).decode()) for n in NAMES}
    for n in NAMES:ast.parse((B/n).read_bytes(),feature_version=(3,8))
    original=json.loads((B/'configs/tmr69d-evaluator-train-spec.json').read_bytes())
    summary=json.loads(pathlib.Path(a.document_summary).read_bytes()) if a.document_summary else None
    if a.action=='enable':
        assert summary['status']=='USER_TMR100_ENDPOINT_FRESH_DOCUMENT_VERIFIED' and summary['findings']==[]
        assert summary['run']=='user_tmr100_control_document_v2' and summary['actual_exit_code']==0
        assert summary['protocol_sha256']==sources['scripts/tmr100_endpoint_protocol.py']['sha256']
        assert summary['entrypoint_sha256']==sources['scripts/watch_tmr100_endpoint.py']['sha256']
        assert summary['actual_production_signals']==0 and summary['actual_fixture_only_signals']==1
    code=r'''
import ast,base64,fcntl,json,os,pathlib,shutil,subprocess,sys,time
r=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from filtered15000_vae_completion import owned,read,sha
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
action=ACTION_LITERAL;role=ROLE_LITERAL;run=RUN_LITERAL;bundle=BUNDLE_LITERAL
parent=owned(r/'evaluator_tmr_20261029');sp=r/'configs/tmr100-user-control-spec.json'
original_sp=r/'configs/tmr69d-evaluator-train-spec.json';orig=read(original_sp)
assert sha(original_sp)=='2196a765dbaa61ba7265b9d798f99fe5c094a502fd34bd47869f0fe141360b13'
assert orig==ORIGINAL_LITERAL
assert sha(r/'scripts/train_tmr69d_evaluator.py')=='04014d8f2e657e9c48827edd7fae817c3f01362d97dcdbd3a0e3dea092c604c2'
if action=='prepare':
 assert not sp.exists()
 for name,item in bundle.items():
  path=r/name;assert not path.exists();raw=base64.b64decode(item['raw']);assert sha_local(raw)==item['sha256']
  ast.parse(raw,feature_version=(3,8))
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  assert sha(owned(path))==item['sha256']
 from tmr100_endpoint_protocol import identity
 producer=identity(416293);supervisor=identity(416285)
 assert producer['start_ticks']=='1441791443' and str(r/'scripts/train_tmr69d_evaluator.py') in producer['argv']
 assert supervisor['start_ticks']=='1441791429' and str(r/'scripts/supervise_tmr69d_evaluator_training.py') in supervisor['argv']
 rec=read(parent/orig['run']/'receipt.json');assert rec['status']=='RUNNING_FRESH_TMR69D_EVALUATOR_TRAINING'
 bound=dict(orig['bound_sources'])
 for name in bundle:bound[name]=sha(r/name)
 for name in ['scripts/vae100k_control_protocol.py','scripts/quick_pidfd.py','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py']:bound[name]=sha(r/name)
 s=dict(schema='textop.user_tmr100_endpoint.v1',control_authorized=False,run=run,target_epochs=100,target_optimizer_updates=101500,
  original_run=orig['run'],original_spec_sha256=sha(original_sp),original_entry_sha256=orig['executor_sha256'],
  original_supervisor_sha256=orig['supervisor_sha256'],original_runtime_binding_sha256=orig['runtime_binding_sha256'],
  accepted_input_signature_sha256=orig['accepted_input_signature_sha256'],
  checkpoint_allowed_root='/home/ouyanghao',checkpoint_path=orig['checkpoint_dir']+'/tmr_epoch_100.pth',checkpoint_upper_bytes=201326592,
  producer=producer,producer_supervisor=supervisor,entrypoint_sha256=sha(r/'scripts/watch_tmr100_endpoint.py'),
  supervisor_sha256=sha(r/'scripts/supervise_tmr100_endpoint.py'),bound_sources=bound,
  author_fixture_binding=None,document_fixture_binding=None,document_summary_sha256=None,
  original_numerical_source_spec_and_LR_unchanged=True,signal_policy='one bound pidfd SIGINT after real100 epoch checkpoint save and production exact reload')
 with sp.open('x') as f:json.dump(s,f,indent=2);f.flush();os.fsync(f.fileno())
 result=dict(status='TMR100_ENDPOINT_PREPARED_DISABLED',spec=s,spec_sha256=sha(sp),
  observed_completed_epochs=rec['completed_epochs'],observed_optimizer_updates=rec['formal_optimizer_updates'],
  actual_production_signals=0,model_loads=0)
else:
 assert sha(sp)==EXPECTED_SPEC_SHA_LITERAL;s=read(sp)
 for name,item in bundle.items():assert sha(r/name)==item['sha256']
 for name,value in s['bound_sources'].items():assert sha(owned(r/name))==value
 from tmr100_endpoint_protocol import verify_identity,identity
 if action=='fixture':
  assert not s['control_authorized'] and role in ['author','document']
  name='user_tmr100_control_'+role+'_v2';folder=parent/name;log=parent/(name+'_supervision')
  assert not folder.exists();log.mkdir(mode=0o700,exist_ok=False)
  cmd=['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/usr/bin/python3','-B',str(r/'scripts/test_tmr100_endpoint.py'),'--out',str(folder)]
  proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',
   lambda pid:sample_resources(r,pid,1024**3,2*1024**3),120,15)
  receipt=read(folder/'receipt.json') if (folder/'receipt.json').exists() else {}
  assert proof['supervision_passed'] and receipt['actual_production_signals']==0 and receipt['actual_fixture_signals']==1
  result=dict(status='TMR100_ENDPOINT_CPU_FIXTURE_CLOSED',run=name,actual_exit_code=proof['child_exit_code'],
   fixture_receipt=receipt,fixture_receipt_sha256=sha(folder/'receipt.json'),process=proof,
   process_exit_sha256=sha(log/'process_exit.json'),resources_sha256=sha(log/'resources.jsonl'),
   protocol_sha256=sha(r/'scripts/tmr100_endpoint_protocol.py'),entrypoint_sha256=sha(r/'scripts/watch_tmr100_endpoint.py'),spec_sha256=sha(sp))
  with (folder/'accepted.json').open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
 elif action=='inspect_fixture':
  assert role in ['author','document'];name='user_tmr100_control_'+role+'_v1';folder=parent/name;log=parent/(name+'_supervision')
  result=dict(status='TMR100_FIXTURE_READ_ONLY_RECONCILIATION',run=name,files={})
  for key,path in [('process',log/'process_exit.json'),('receipt',folder/'receipt.json'),('accepted',folder/'accepted.json')]:
   if path.exists():result['files'][key]=read(path)
  if (log/'child.log').exists():result['child_log_tail']=(log/'child.log').read_text(errors='replace')[-5000:]
 elif action=='enable':
  assert not s['control_authorized'];summary=SUMMARY_LITERAL
  bindings={}
  for role in ['author','document']:
   name='user_tmr100_control_'+role+'_v2';folder=parent/name;log=parent/(name+'_supervision');proof=read(log/'process_exit.json');receipt=read(folder/'receipt.json')
   assert proof['supervision_passed'] and proof['child_reaped'] and proof['child_exit_code']==0 and proof['sample_errors']==[]
   rows=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
   assert rows[-1]['phase']=='after_exit' and rows[-1]['owned_group_processes']==[] and rows[-1]['owned_group_gpu_memory']==[]
   assert len(rows)==proof['samples_written']
   assert receipt['actual_production_signals']==0 and receipt['production_checkpoint_reads']==receipt['model_loads']==receipt['new_GPU_contexts']==0
   assert receipt['actual_fixture_signals']==1 and receipt['fixture_child_reaped'] and receipt['direct_streams_final']
   bindings[role]=dict(run=name,receipt_sha256=sha(folder/'receipt.json'),process_exit_sha256=sha(log/'process_exit.json'),resources_sha256=sha(log/'resources.jsonl'))
  assert summary['fixture_receipt_sha256']==bindings['document']['receipt_sha256']
  assert summary['protocol_sha256']==sha(r/'scripts/tmr100_endpoint_protocol.py') and summary['entrypoint_sha256']==s['entrypoint_sha256']
  verify_identity(s['producer']);verify_identity(s['producer_supervisor'])
  path=r/'receipts/tmr100_control_spec_before_enable.json'
  with path.open('xb') as f:f.write(sp.read_bytes());f.flush();os.fsync(f.fileno())
  docraw=(json.dumps(summary,indent=2)+'\n').encode();doc=r/'receipts/tmr100_control_fresh_document_verified.json'
  with doc.open('xb') as f:f.write(docraw);f.flush();os.fsync(f.fileno())
  s.update(control_authorized=True,author_fixture_binding=bindings['author'],document_fixture_binding=bindings['document'],document_summary_sha256=sha(doc))
  raw=(json.dumps(s,indent=2)+'\n').encode();partial=sp.with_name(sp.name+'.enable.partial')
  with partial.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  partial.replace(sp);result=dict(status='TMR100_ENDPOINT_ENABLED_NOT_DISPATCHED',spec=s,spec_sha256=sha(sp),actual_production_signals=0)
 elif action=='launch':
  assert s['control_authorized'] and s['author_fixture_binding'] and s['document_fixture_binding']
  verify_identity(s['producer']);verify_identity(s['producer_supervisor'])
  prefix='tmr100_control_'+run;paths=[r/'receipts'/(prefix+x) for x in ['_driver.py','_driver.json','_outer_exit.json']]
  assert not any(x.exists() for x in paths) and not (parent/run).exists()
  lock=(r/'receipts/user_tmr100_control.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  resources=sample_resources(r,None,1024**3,2*1024**3)
  outer=paths[2];cmd=['taskset','-c','0-3','/usr/bin/python3','-B',str(r/'scripts/supervise_tmr100_endpoint.py'),'--run',run]
  text='import subprocess,pathlib,json,os\np=subprocess.run('+repr(cmd)+')\nwith pathlib.Path('+repr(str(outer))+').open("x") as f:json.dump({"actual_returncode":p.returncode},f);f.flush();os.fsync(f.fileno())\nraise SystemExit(p.returncode)\n'
  with paths[0].open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
  lock.close()
  with (parent/(run+'_outer.log')).open('xb') as f:q=subprocess.Popen(['/usr/bin/python3','-B',str(paths[0])],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=r,start_new_session=True)
  result=dict(status='TMR100_ENDPOINT_DISPATCHED_NOT_ACCEPTED',run=run,driver_pid=q.pid,spec_sha256=sha(sp),resources=resources,actual_production_signals=0)
  with paths[1].open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
 else:
  assert action=='collect';prefix='tmr100_control_'+run;files={};hashes={}
  for name,path in dict(receipt=parent/run/'receipt.json',exit=r/'receipts'/(prefix+'_exit.json'),process=r/'receipts'/(prefix+'_process_exit.json'),outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
   if path.exists():files[name]=read(path);hashes[name]=sha(path)
  workers=[]
  for proc in pathlib.Path('/proc').iterdir():
   if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
   try:
    if proc.stat().st_uid!=1013:continue
    argv=(proc/'cmdline').read_bytes().decode().split('\0')
    if any(str(r/'scripts'/n) in argv for n in ['watch_tmr100_endpoint.py','supervise_tmr100_endpoint.py']):workers.append(identity(int(proc.name)))
   except (FileNotFoundError,ProcessLookupError):pass
  common=(r/'receipts/full_library_capacity.lock').open('a+')
  try:fcntl.flock(common,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
  except BlockingIOError:free=False
  common.close()
  closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and not workers and free
  if closed:
   assert files['receipt']['status']=='USER_TMR100_EPOCH_STOP_ACCEPTED_ORIGINAL_EXITS_RETAINED' and files['receipt']['signals_sent']==1
   assert files['receipt']['output_writers_closed'] and files['receipt']['selected_epochs']==100 and files['receipt']['selected_optimizer_updates']==101500
   assert hashes['receipt']==files['exit']['receipt_sha256'] and hashes['process']==files['exit']['process_exit_sha256']
   resources=r/'logs'/(prefix+'_resources.jsonl');rows=[json.loads(x) for x in resources.read_text().splitlines()]
   assert sha(resources)==files['exit']['resource_sha256'] and len(rows)==files['exit']['samples_written']==files['process']['samples_written']
   assert rows[-1]['phase']=='after_exit' and rows[-1]['owned_group_processes']==[]
  result=dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,
   common_lock_free=free,closed_execution_verified=bool(closed),model_loads=0)
print(json.dumps(result))
'''
    code=code.replace('ROOT_LITERAL',repr(ROOT)).replace('ACTION_LITERAL',repr(a.action)).replace('ROLE_LITERAL',repr(a.role)).replace('RUN_LITERAL',repr(RUN)).replace('BUNDLE_LITERAL',repr(sources)).replace('ORIGINAL_LITERAL',repr(original)).replace('EXPECTED_SPEC_SHA_LITERAL',repr(sha(spec_path) if spec else None)).replace('SUMMARY_LITERAL',repr(summary))
    code=code.replace('sha_local(raw)',"__import__('hashlib').sha256(raw).hexdigest()")
    ast.parse(code,feature_version=(3,8))
    result=json.loads(recorded_remote(code,out/(a.evidence+'_client.json'),timeout=180))
    persist(out/(a.evidence+'.json'),result)
    if a.action in ['prepare','enable']:
        spec_path.write_bytes((json.dumps(result['spec'],indent=2)+ ('\n' if a.action=='enable' else '')).encode('utf-8'))
        assert sha(spec_path)==result['spec_sha256']
    if a.action=='collect':
        (out/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
