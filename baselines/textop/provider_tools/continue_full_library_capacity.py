"""Finish one new KIT backup audit, then one full-library author capacity run.

Owns neither GMR nor its backup/FK queue. No dataset/checkpoint deletion and no
formal policy/generator launch. All waits depend on actual existing receipts.
"""
import argparse,base64,datetime,hashlib,json,os,pathlib,subprocess,sys,time
from session import REPO,SSH,remote

B=REPO/'baselines/textop';TASK=REPO.parent.parent/'textop_reproduction_20261024'
DEST=B/'results/full_library_capacity_20261026';STATE=DEST/'continuation_v1.json'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RUN='full_library_author_v1';AUDIT='after_kit_v1'
EXPECTED={
 'scripts/full_library_capacity_probe.py':'7d25ad8238977fa6397080732c7ddade1c7414305ae2f254fbe42d5326357e90',
 'scripts/supervise_full_library_capacity.py':'4dcda5dfc9b81efc3a6719229505e1ad3cb731d52bd7beda836d455d22c52846',
 'scripts/full_library_inputs.py':'4b5784cf49e9b010de84be4194071cccb8f23a990fa179bb70b1157320265068',
 'configs/full-library-capacity-spec.json':'d720d2b89071956e45faac6d0dce19ce334e47ff07eb7f96802bf17cd442214d'}

def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def read(path):
 for attempt in range(10):
  try:return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
  except json.JSONDecodeError:time.sleep(1)
 raise RuntimeError('Repeated incomplete JSON receipt')
def state(status,**values):
 data=read(STATE) if STATE.exists() else {}
 data.update(status=status,pid=os.getpid(),time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             physical_filter_executed=False,dataset_accepted=False,formal_filter_training_started=False,full_generator_training_started=False,**values)
 tmp=STATE.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');tmp.replace(STATE)
 print(json.dumps(data),flush=True)
def run_local(name,args=()):
 env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
 proc=subprocess.run([sys.executable,str(B/'provider_tools'/name),*args],capture_output=True,text=True,encoding='utf-8',env=env,timeout=900)
 # Full diagnostics stay in the private task workspace until reviewed.
 (TASK/('full_library_'+name.replace('.py','')+'.stdout.log')).write_text(proc.stdout,encoding='utf-8')
 (TASK/('full_library_'+name.replace('.py','')+'.stderr.log')).write_text(proc.stderr,encoding='utf-8')
 if proc.returncode:raise RuntimeError(name+' exited '+str(proc.returncode)+'; local diagnostics retained')
 return proc
def snapshot():
 return json.loads(remote(r'''
import pathlib,json,shutil,subprocess,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
paths={'export':pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references/export_kit_v1/receipt.json'),
       'exit':r/'receipts/tracker_job_export_kit_v1_exit.json'}
out={}
for key,path in paths.items():
 for attempt in range(10):
  try:out[key]=json.loads(path.read_text()) if path.exists() else {};break
  except json.JSONDecodeError:time.sleep(1)
out['data_free_bytes']=shutil.disk_usage(r).free
out['mem_available_bytes']=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip()
print(json.dumps(out))
''',timeout=90))

def main(check_only=False):
 assert os.name=='nt'
 for name,digest in EXPECTED.items():assert sha(B/name)==digest,'Declared draft changed; inspect before continuation'
 proof={'queue_status':read(B/'results/data_preparation_20261025/kit_indexed_queue.json')['status'],
        'audit_output_exists':(B/'results/preprocessing_backup_audit_20261025'/AUDIT).exists(),
        'continuation_exists':STATE.exists(),'server':snapshot(),'no_job_started':True}
 if check_only:print(json.dumps(proof));return
 import msvcrt
 lock=(TASK/'full_library_capacity_continuation.lock').open('a+b')
 if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
 lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 assert not STATE.exists() and not proof['audit_output_exists'],'Existing continuation/audit retained; do not duplicate'
 state('WAITING_FOR_EXISTING_KIT_DURABLE_BACKUP',audit_run=AUDIT,capacity_run=RUN)
 try:
  for _ in range(8640):
   queue=read(B/'results/data_preparation_20261025/kit_indexed_queue.json')
   if queue['status']=='PREPROCESSING_BACKED_UP_FILTER_PENDING':break
   if queue['status'].startswith('STOPPED') or 'PARTIAL_FAILURE' in queue['status']:raise RuntimeError('Existing KIT coordinator reported a failure; no duplicate or repair launched')
   time.sleep(30)
  else:raise TimeoutError('KIT persistent backup not complete in72h; original workers retained')
  assert queue['progress']['completed_sequences']==2513 and queue['progress']['failed_sequences']==0
  state('RUNNING_NEW_KIT_PERSISTENT_CONTENT_AUDIT')
  proc=run_local('audit_preprocessing_backups.py',['--run',AUDIT])
  auditpath=B/'results/preprocessing_backup_audit_20261025'/AUDIT/'receipt.json';audit=read(auditpath)
  assert audit['status']=='PREPROCESSING_ALL_AVAILABLE_SOURCES_BACKED_UP_VERIFIED_FILTER_PENDING'
  assert audit['verified_sources']==8278 and audit['preprocessing_complete'] is True and audit['output_writers_closed'] is True
  assert audit['dataset_accepted'] is False and audit['physical_filter_executed'] is False
  (DEST/'after_kit_audit_execution.json').write_text(json.dumps({'exit_code':proc.returncode,'receipt_sha256':sha(auditpath),'run':AUDIT},indent=2),encoding='utf-8')
  state('PREPROCESSING_ALL_AVAILABLE_BACKED_UP_WAITING_FK',preprocessing_complete=True,verified_sources=8278,backup_receipt_sha256=sha(auditpath))
  for script in ['collect_preprocessing_audit.py','collect_data_preparation.py','collect_tracker_progress.py','publish.py']:run_local(script)
  for _ in range(8640):
   snap=snapshot();ref=snap['export'];exitproof=snap['exit']
   if ref.get('status')=='FAILED' or (exitproof and exitproof.get('child_exit_code')!=0):raise RuntimeError('Original KIT FK export failed; all records retained')
   if ref.get('status')=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED' and exitproof.get('child_exit_code')==0 and exitproof.get('timed_out') is False:break
   time.sleep(30)
  else:raise TimeoutError('Existing KIT FK queue not complete in72h')
  assert ref['exported_source_sequences']==2513 and ref['output_writers_closed'] is True
  cap=read(B/'configs/full-library-capacity-spec.json')
  assert snap['data_free_bytes']>cap['minimum_free_data_disk_gib']*1024**3
  assert snap['mem_available_bytes']>cap['minimum_available_host_ram_gib']*1024**3
  gpu={int(x.split(',')[0]):int(x.split(',')[1])*1024**2 for x in snap['gpu'].splitlines()};assert gpu[1]>cap['minimum_free_gpu_gib']*1024**3
  (DEST/'disabled_spec_before_author.json').write_text(json.dumps(cap,indent=2),encoding='utf-8')
  cap.update(launch_enabled=True,launch_blockers=[],persistent_backup_acceptance={'receipt':ROOT+'/receipts/preprocessing_after_kit_v1.json','sha256':sha(auditpath)})
  data=base64.b64encode(auditpath.read_bytes()).decode();capbytes=(json.dumps(cap,indent=2)+'\n').encode();encodedcap=base64.b64encode(capbytes).decode()
  activated=json.loads(remote('import pathlib,json,hashlib,base64\nr=pathlib.Path('+repr(ROOT)+')\nexpected=json.loads('+repr(json.dumps(EXPECTED))+')\nfor n,digest in expected.items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==digest\nassert not (r/"tracker_validation/'+RUN+'").exists()\na=r/"receipts/preprocessing_after_kit_v1.json";assert not a.exists();a.write_bytes(base64.b64decode('+repr(data)+'))\nc=r/"configs/full-library-capacity-spec.json";c.write_bytes(base64.b64decode('+repr(encodedcap)+'))\nprint(json.dumps({"receipt_sha256":hashlib.sha256(a.read_bytes()).hexdigest(),"capacity_spec_sha256":hashlib.sha256(c.read_bytes()).hexdigest()}))'))
  assert activated['receipt_sha256']==sha(auditpath) and activated['capacity_spec_sha256']==hashlib.sha256(capbytes).hexdigest()
  (B/'configs/full-library-capacity-spec.json').write_bytes(capbytes)
  state('FULL_LIBRARY_AUTHOR_CAPACITY_GATE_ACTIVATED',activation=activated,resource_snapshot=snap)
  run_local('publish.py')
  cmd='python3 '+ROOT+'/scripts/supervise_full_library_capacity.py --run '+RUN
  state('RUNNING_FULL_LIBRARY_AUTHOR_CAPACITY',command=cmd)
  with (TASK/'full_library_author_stdout.log').open('xb') as stdout,(TASK/'full_library_author_stderr.log').open('xb') as stderr:
   result=subprocess.run(SSH+[cmd],stdout=stdout,stderr=stderr,timeout=1950)
  evidence=json.loads(remote('import pathlib,json\nr=pathlib.Path('+repr(ROOT)+')\nprint(json.dumps({"receipt":json.loads((r/"tracker_validation/'+RUN+'/receipt.json").read_text()),"exit":json.loads((r/"receipts/full_library_capacity_'+RUN+'_exit.json").read_text())}))'))
  out=DEST/RUN;out.mkdir(exist_ok=False)
  for key,value in evidence.items():(out/(key+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
  (out/'ssh_execution.json').write_text(json.dumps({'exit_code':result.returncode,'command':cmd},indent=2),encoding='utf-8')
  assert result.returncode==0 and evidence['exit']['child_exit_code']==0 and evidence['exit']['timed_out'] is False
  assert evidence['receipt']['status']=='TEXTOP_FULL_LIBRARY_8192_CAPACITY_PROBE_PASSED'
  state('FULL_LIBRARY_AUTHOR_VERIFIED_DOCUMENT_PENDING',author_capacity_passed=True,independent_document_pending=True)
  run_local('publish.py')
 except BaseException as error:
  state('STOPPED_WITH_ERROR_ALL_DATA_RETAINED',error=repr(error));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');main(p.parse_args().check_only)
