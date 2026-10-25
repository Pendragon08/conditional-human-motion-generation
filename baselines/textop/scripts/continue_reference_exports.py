"""One serial, locked FK continuation; never retarget/train/delete or back up."""
import fcntl,hashlib,json,os,pathlib,shutil,subprocess,time,traceback
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RAM=pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
assert os.getuid()==1013
lock=(ROOT/'receipts/reference_export_queue_v1.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
spec_path=ROOT/'configs/reference-export-queue.json';spec=json.loads(spec_path.read_text())
state_path=ROOT/'receipts/reference_export_queue_v1.json';assert not state_path.exists(),'Existing queue evidence retained; inspect instead of duplicate restart'
state={'status':'STARTING','pid':os.getpid(),'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),
 'completed_cohorts':[],'physical_filter_executed':False,'dataset_accepted':False,'full_generator_training_started':False}
def save(status,**values):
 state.update(status=status,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
 tmp=state_path.with_suffix('.json.partial');tmp.write_text(json.dumps(state,indent=2));tmp.replace(state_path)
def own_sdk_active():
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=1013:continue
   cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if str(ROOT/'scripts') in cmd and '--headless' in cmd and '--device' in cmd:return True
 return False
try:
 save('WAITING_FOR_COMPLETED_PREPARATION')
 for cohort in spec['cohorts']:
  folder=RAM/cohort['prepared'];assert RAM in folder.resolve().parents
  started=time.monotonic()
  while True:
   if time.monotonic()-started>spec['maximum_wait_seconds_per_cohort']:raise TimeoutError('Cohort prerequisites did not complete in72h; all data/jobs retained')
   progress_path=folder/'progress.json'
   try:progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
   except json.JSONDecodeError:time.sleep(1);continue
   if progress.get('status')=='PREPROCESSING_FAILED':raise RuntimeError('Preprocessing failed; no reference export started')
   if progress.get('status')!='PREPROCESSING_FINISHED_FILTER_PENDING':
    save('WAITING_FOR_COMPLETED_PREPARATION',cohort=cohort['run'],prepared_sources=progress.get('completed_sequences',0));time.sleep(spec['poll_seconds']);continue
   assert progress['completed_sequences']==cohort['expected_sources'] and progress['failed_sequences']==0
   free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','--id=1'],text=True).strip())*1024**2
   ready=free>spec['minimum_free_gpu_gib']*1024**3 and shutil.disk_usage(ROOT).free>spec['minimum_free_data_gib']*1024**3 and shutil.disk_usage(RAM).free>spec['minimum_free_shm_gib']*1024**3 and not own_sdk_active()
   if not ready:
    save('WAITING_FOR_GPU_OR_STORAGE_RESERVE',cohort=cohort['run'],gpu1_free_bytes=free);time.sleep(spec['poll_seconds']);continue
   break
  assert not (RAM/'tracker_references'/cohort['run']).exists(),'Unexpected output retained; no duplicate launch'
  command=['python3',str(ROOT/'scripts/supervise_tracker_job.py'),'--run',cohort['run'],'--job',cohort['job']]
  with (ROOT/'logs'/('reference_queue_'+cohort['run']+'.log')).open('xb') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   save('RUNNING_REFERENCE_EXPORT',cohort=cohort['run'],supervisor_pid=child.pid,command=command)
   code=child.wait()
  proof=json.loads((ROOT/'receipts'/('tracker_job_'+cohort['run']+'_exit.json')).read_text())
  assert code==0 and proof['child_exit_code']==0 and not proof['timed_out'] and proof['numerical_status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
  receipt=json.loads((RAM/'tracker_references'/cohort['run']/'receipt.json').read_text())
  assert receipt['exported_source_sequences']==cohort['expected_sources'] and receipt['exported_trajectories']==2*cohort['expected_sources']
  state['completed_cohorts'].append({'run':cohort['run'],'sources':receipt['exported_source_sequences'],'frames':receipt['exported_frames'],'private_manifest_sha256':receipt['private_manifest_sha256']})
  save('COHORT_REFERENCE_EXPORT_FINISHED')
 save('ALL_DECLARED_REFERENCE_EXPORTS_FINISHED')
except BaseException as error:
 save('STOPPED_WITH_ERROR_ALL_DATA_RETAINED',error=repr(error),traceback=traceback.format_exc());raise
