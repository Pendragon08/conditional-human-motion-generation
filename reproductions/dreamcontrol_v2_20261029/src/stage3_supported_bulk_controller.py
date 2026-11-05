"""Generate/audit/fit/native-admit rolling cohorts to500 per supported task.

Adopts the already-running first32 producer and reuses its two solved states.
Never starts training, repeats completed cohorts or signals unrelated jobs.
"""
import ast,collections,fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'bulk_v1';S=R/'stage3_pick29_20261103';D=R/'stage2_prior29_20261103';P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
sys.path[:0]=[str(S/'src'),str(R/'stage3_supermimic29_20261103/src')]
from stage3_pick500_replacement_queue import snapshot
from stage3_pick_parallel_controller import resources,stop_group
TASKS=['pour','bimanual_box_lift','box_push_pull'];BASE={'pour':11000000,'bimanual_box_lift':12000000,'box_push_pull':13000000};FLOOR=8*1024**3+456*1024**2
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines()]
def atomic(p,j):
 t=p.with_suffix('.bulk.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
lock=(W/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (W/'controller_status.json').exists();assert read(W/'pilot_admission_audit.json')['status']=='six_native_pilots_independent_binding_passed'
sources=[B/'src'/x for x in ['stage3_sample_supported29_batch.py','stage3_supported_task_conditions.py','stage3_supported_bulk_postopt_v1.py','stage3_supported_bulk_audit.py','stage3_supported_native_replay.py']]
bindings={str(p):sha(p) for p in sources};(W/'source_binding.json').open('x').write(json.dumps(bindings,indent=2))
spec=read(D/'env_spec.json');assert spec['spec_hash']=='74af067a';cfg=read(D/'config.json')
sample_env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':':'.join(spec['spec']['module_paths'])};sample_env.pop('CUDA_VISIBLE_DEVICES',None)
tree=ast.parse((S/'src/stage3_postopt_queue.py').read_text());assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='env' for n in x.targets));ns={'R':R,'S':S,'Path':pathlib.Path,'os':os};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'existing PyRoki environment only','exec'),ns);opt_env=ns['env'];opt_env['CUDA_VISIBLE_DEVICES']='0';opt_env['PYTHONDONTWRITEBYTECODE']='1'
native_spec=read(R/'stage3_supermimic29_20261103/runtime_bridge_spec.json');native_env={**os.environ,**native_spec['env']};native_env['PYTHONPATH']=str(B/'src')+':'+native_env['PYTHONPATH'];native_env.pop('CUDA_VISIBLE_DEVICES',None)
audit_env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'};audit_env.pop('CUDA_VISIBLE_DEVICES',None)
library=rows(W/'library_records.jsonl');state={'status':'adopting_first32_producer','pid':os.getpid(),'target_per_task':500,'target_total':1500,'formal_training_started':False,'sampled_new_candidates':{t:0 for t in TASKS},'cohorts_completed':[]};child=None;started=time.monotonic()
def counts():return {t:sum(r['task']==t for r in library) for t in TASKS}
def save():
 state.update(native_admitted_counts=counts(),selected_reference_counts={t:min(500,n) for t,n in counts().items()},elapsed_seconds=time.monotonic()-started);atomic(W/'controller_status.json',state)
def check():
 for p,h in bindings.items():assert sha(p)==h,'bound source changed; preserve job instead of silently continuing'
def wait_space():
 while shutil.disk_usage('/data').free<FLOOR+200*1024**2:
  state.update(status='waiting_disk_before_next_cohort',free_bytes=shutil.disk_usage('/data').free);save();time.sleep(15)
def run(label,command,env,cwd,cohort,completion=None):
 global child
 check();log=(cohort/(label+'.log')).open('x');child=subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 phase=time.monotonic();done_since=None;state.update(status=label,child_pid=child.pid,active_cohort=str(cohort),command=command);save()
 (cohort/(label+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,'command':command,'bindings':bindings},indent=2))
 while child.poll() is None:
  own=resources(child.pid);total=snapshot();state.update(resources=own,aggregate_resources=total);save()
  assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3 and total['owned_gpu_bytes']<=24*1024**3 and total['owned_rss_bytes']<=64*1024**3
  assert shutil.disk_usage('/data').free>=FLOOR and time.monotonic()-phase<7200
  if completion and completion.exists() and read(completion).get('status')=='native_task_replay_completed':
   if done_since is None:done_since=time.monotonic()
   if time.monotonic()-done_since>30:stop_group(child.pid);state['released_own_completed_native_cleanup_group']=True;break
  time.sleep(3)
 code=child.wait();log.close();child=None
 assert code==0 or (completion and completion.exists() and read(completion).get('status')=='native_task_replay_completed'),'phase failed; preserve prefix and inspect original log'
 return time.monotonic()-phase
def process(cohort,task,count,first=False):
 producer=cohort/'candidates';state.update(active_task=task,active_cohort=str(cohort));save();check()
 if first:
  adopt=read(W/'first_producer_launch.json');assert pathlib.Path(adopt['cohort'])==cohort
  began=time.monotonic()
  while True:
   j=read(cohort/'producer_guard.json')
   if j['status']=='failed_preserved':raise RuntimeError('adopted producer failed; preserve actual prefix')
   if j['status']=='batch_candidates_generated_postoptimization_pending':break
   assert time.monotonic()-began<7200;state.update(status='waiting_adopted_first32_producer',adopted_producer=j);save();time.sleep(10)
 else:
  cohort.mkdir(parents=True,exist_ok=False)
  start=BASE[task]+state['sampled_new_candidates'][task]
  run('sampling_candidates',['nice','-n','10',P,'-u',str(B/'src/stage3_sample_supported29_batch.py'),'--task',task,'--count',str(count),'--start-index',str(start),'--output-dir',str(producer),'--batch-size','2'],sample_env,cfg['trajgen'],cohort)
 assert read(producer/'report.json')['generated_candidates']==count;state['sampled_new_candidates'][task]+=count
 run('audit_raw_candidates',['nice','-n','10',P,'-u',str(B/'src/stage3_supported_bulk_audit.py'),'--mode','raw','--producer',str(producer),'--output',str(cohort/'raw_audit.json')],audit_env,str(B),cohort)
 prefix=[];exclude=[]
 if first:
  witness=W/'optimization_witness';began=time.monotonic()
  while True:
   j=read(witness/'status.json')
   if j['status']=='failed_preserved':raise RuntimeError('production optimizer witness failed; inspect existing two states')
   if j['status']=='production_optimizer_first_two_completed_reuse_required':break
   assert time.monotonic()-began<1800;time.sleep(5)
  prefix=rows(witness/'postopt/records.jsonl');assert [r['candidate_id'] for r in prefix]==[11000000,11000001]
  exclude=['--exclude-records',str(witness/'postopt/records.jsonl')]
 remaining=count-len(prefix)
 if remaining:
  run('postopt_generated29',['nice','-n','10',P,'-u',str(B/'src/stage3_supported_bulk_postopt_v1.py'),'--records',str(producer/'records.jsonl'),'--output-dir',str(cohort/'postopt'),'--max-candidates',str(remaining)]+exclude,opt_env,str(B),cohort)
  optimized=prefix+rows(cohort/'postopt/records.jsonl')
 else:optimized=prefix
 assert len(optimized)==count
 combined=cohort/'combined_robot_records.jsonl';combined.open('x').write(''.join(json.dumps(r)+'\n' for r in sorted(optimized,key=lambda x:x['candidate_id'])))
 run('audit_robot_states',['nice','-n','10',P,'-u',str(B/'src/stage3_supported_bulk_audit.py'),'--mode','robot','--producer',str(producer),'--robot-records',str(combined),'--output',str(cohort/'robot_audit.json')],audit_env,str(B),cohort)
 selected=[r for r in optimized if r['passed_requested_kinematic_checks']];accepted=[]
 if selected:
  command=[native_spec['python'],'-u',str(B/'src/stage3_supported_native_replay.py'),'--headless','--device','cuda:0','--task',task,'--output-dir',str(cohort/'native')]
  for r in selected:command+=['--candidate-dir',str(pathlib.Path(r['output']).parent)]
  run('native_task_scene_replay',command,native_env,native_spec['cwd'],cohort,cohort/'native/report.json')
  run('audit_native_bindings',['nice','-n','10',P,'-u',str(B/'src/stage3_supported_bulk_audit.py'),'--mode','native','--producer',str(producer),'--robot-records',str(combined),'--native-dir',str(cohort/'native'),'--output',str(cohort/'native_audit.json')],audit_env,str(B),cohort)
  accepted=read(cohort/'native_audit.json')['accepted_references']
 duplicates=[];added=0
 for r in accepted:
  if any(any(old[k]==r[k] for k in ['candidate_id','state_fingerprint','initial_noise_sha256','feature_content_sha256']) for old in library):duplicates.append(r['candidate_id']);continue
  with (W/'library_records.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  library.append(r);added+=1
 summary={'status':'cohort_completed','task':task,'candidates':count,'kinematic_passed':len(selected),'native_eligible':len(accepted),'unique_admitted':added,'duplicates':duplicates,'raw_audit_sha256':sha(cohort/'raw_audit.json'),'robot_audit_sha256':sha(cohort/'robot_audit.json'),'native_audit_sha256':sha(cohort/'native_audit.json') if accepted else None}
 (cohort/'summary.json').open('x').write(json.dumps(summary,indent=2));state['cohorts_completed'].append(summary);save()
 assert added>0,'zero-admission cohort: diagnose the preserved results before spending on more samples'
try:
 save();process(W/'cohorts/pour_0001','pour',32,True);numbers={'pour':1,'bimanual_box_lift':0,'box_push_pull':0}
 while any(n<500 for n in counts().values()):
  for task in TASKS:
   needed=500-counts()[task]
   if needed<=0:continue
   cap=1000-state['sampled_new_candidates'][task];assert cap>0,'bounded1000-candidate/task budget exhausted; diagnose quality, no blind retry'
   count=min(32,needed+4,cap);wait_space();numbers[task]+=1;process(W/'cohorts'/(task+'_'+str(numbers[task]).zfill(4)),task,count)
 for task in TASKS:
  selected=sorted([r for r in library if r['task']==task],key=lambda r:r['candidate_id'])[:500];assert len(selected)==500
  for r in selected:assert sha(r['path'])==r['sha256'] and sha(r['native_proof'])==r['native_proof_sha256']
  (W/(task+'_500_scene_references.json')).open('x').write(json.dumps({'task':task,'body_dof':29,'reference_fps':20,'reference_count':500,'references':selected,'scope':'scene-admitted reference subset; final mixed loader/statistics and training resource gate remain'},indent=2))
 state.update(status='all1500_scene_references_admitted_final_mixed_loader_pending',formal_training_started=False,next='independent1500 selection audit, combine with existingPick500, native20Hz2000-library loading and final dataset receipt; no training until actual four-library gate passes');save()
 (W/'completion.json').open('x').write(json.dumps(state,indent=2))
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child and child.poll() is None:stop_group(child.pid)
 raise
finally:lock.close()
