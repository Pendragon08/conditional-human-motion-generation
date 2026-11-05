"""Wait for real1500 admission, then finalize/load2000 on CPU. No training."""
import fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
B=R/'stage3_supported_references_20261104';W=B/'bulk_v1';D=R/'stage3_supermimic29_20261103'
N=D/'native_universal_20261104';S=R/'stage3_pick29_20261103';J=B/'mixed2000_completion_v1';FINAL=B/'mixed2000_final_v1'
P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
FLOOR=8*1024**3+456*1024**2
sys.path[:0]=[str(D/'src'),str(S/'src')]
from stage3_pick_parallel_controller import resources,stop_group
from stage3_pick500_replacement_queue import snapshot
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def atomic(p,j):
 t=p.with_suffix('.mixed.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(p)
J.mkdir(exist_ok=False);lock=(J/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sources=[B/'src/stage3_supported_mixed2000_prepare.py',B/'src/stage3_supported_native_library_check.py']
bindings={str(p):sha(p) for p in sources}
(J/'source_binding.json').open('x').write(json.dumps(bindings,indent=2))
state={'status':'waiting_all1500_scene_admissions','pid':os.getpid(),'formal_training_started':False,'GPU_work':False,'output':str(FINAL)}
child=None;began=time.monotonic()
def save():state['elapsed_seconds']=time.monotonic()-began;atomic(J/'status.json',state)
def check():
 for p,h in bindings.items():assert sha(p)==h,'completion input source changed; preserve prefix'
def run(label,command,expected,path,env,cwd):
 global child
 check();log=(J/(label+'.log')).open('x')
 child=subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (J/(label+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,'command':command,'bound_sources':bindings},indent=2))
 phase=time.monotonic();state.update(status=label,child_pid=child.pid);save()
 while child.poll() is None:
  own=resources(child.pid);total=snapshot();state.update(resources=own,aggregate_resources=total);save()
  assert own['gpu_bytes']==0 and own['rss_bytes']<=32*1024**3,'CPU data-finalization memory contract'
  assert total['owned_gpu_bytes']<=24*1024**3 and total['owned_rss_bytes']<=64*1024**3
  assert shutil.disk_usage('/data').free>=FLOOR and time.monotonic()-phase<7200
  time.sleep(3)
 code=child.wait();log.close();child=None
 assert code==0 and read(path)['status']==expected,'finalization stage failed; inspect preserved prefix'
 return read(path)
try:
 save()
 while not (W/'completion.json').exists():
  check();bulk=read(W/'controller_status.json');state['bulk_status']=bulk['status'];state['native_admitted_counts']=bulk['native_admitted_counts'];save()
  assert bulk['status']!='failed_preserved','upstream admission failed; preserve sources and do not finalize partial library'
  time.sleep(15)
 completion=read(W/'completion.json');assert completion['status']=='all1500_scene_references_admitted_final_mixed_loader_pending'
 assert all(completion['native_admitted_counts'][t]>=500 for t in ['pour','bimanual_box_lift','box_push_pull'])
 while shutil.disk_usage('/data').free<FLOOR+150*1024**2:
  state.update(status='waiting_final_input_disk_reserve',free_bytes=shutil.disk_usage('/data').free);save();time.sleep(15)
 env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
 run('prepare_final2000',['nice','-n','10',P,'-B','-u',str(sources[0]),'--output-dir',str(FINAL)],'final2000_inputs_native_loader_pending',FINAL/'status.json',env,str(B))
 native_python=read(N/'environment_spec.json')['spec']['python'];loader=FINAL/'native_loader_check_v1'
 run('load_check_final2000',['nice','-n','10',native_python,'-I','-B','-u',str(sources[1]),'--prepared-dir',str(FINAL),'--output-dir',str(loader)],'all2000_native_library_passed',loader/'status.json',env,str(N/'source'))
 manifest=FINAL/'native_motion_manifest.json';m=read(manifest);result=read(loader/'report.json')
 assert m['reference_count']==result['reference_count']==2000 and m['per_task']==result['per_task']=={t:500 for t in ['pick','pour','bimanual_box_lift','box_push_pull']}
 assert not m['preview'] and result['manifest_sha256']==sha(manifest) and result['all_clips_future_goal_and_boundary_passed']
 receipt={'status':'four_task2000_reference_dataset_completed','reference_count':2000,'per_task':m['per_task'],
  'manifest':str(manifest),'manifest_sha256':sha(manifest),'native_loader_report':str(loader/'report.json'),'native_loader_report_sha256':sha(loader/'report.json'),
  'native_motion_dir':str(FINAL/'motion_csv20'),'bulk_completion':str(W/'completion.json'),'bulk_completion_sha256':sha(W/'completion.json'),
  'source_bindings':bindings,'data_ready_for_native_body_tracker':True,'formal_training_started':False,
  'source_fps':20,'native_control_fps':50,'body_dof':29,'native_reference_frames':result['native50Hz_frames'],
  'hand_and_object_metadata_retained':True,'scope':'four task body reference input acceptance; not policy tracking or object task success',
  'next':'notify dataset completion; native formal trainer requires its own budget, resource protection and independent evaluation plan',
  'elapsed_seconds':time.monotonic()-began}
 (FINAL/'completion_receipt.json').open('x').write(json.dumps(receipt,indent=2,allow_nan=False))
 state.update(status=receipt['status'],receipt=str(FINAL/'completion_receipt.json'),receipt_sha256=sha(FINAL/'completion_receipt.json'));save()
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child and child.poll() is None:stop_group(child.pid)
 raise
finally:lock.close()
