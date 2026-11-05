"""Sequential bounded pilots on GPU0; never start training or touch Pick PPO."""
import ast,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';D=R/'stage2_prior29_20261103';B=R/'stage3_supported_references_20261104';W=B/'pilot_v2';P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
sys.path[:0]=[str(S/'src'),str(R/'stage3_supermimic29_20261103/src')]
from stage3_pick500_replacement_queue import snapshot
from stage3_pick_parallel_controller import resources,stop_group
assert not W.exists();W.mkdir();(W/'condition_preparation_report.json').write_bytes((B/'pilot_v1/condition_preparation_report.json').read_bytes())
assert not (W/'status.json').exists();prep=json.loads((W/'condition_preparation_report.json').read_text());assert prep['status']=='task_condition_plans_checked_not_generated'
spec=json.loads((D/'env_spec.json').read_text());assert spec['spec_hash']=='74af067a'
env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':':'.join(spec['spec']['module_paths'])}
env.pop('CUDA_VISIBLE_DEVICES',None)
cfg=json.loads((D/'config.json').read_text());state={'status':'initializing','pid':os.getpid(),'generated_candidates':0,'accepted_references':0,'SuperMimic_training_started':False};child=None;started=time.monotonic()
def save():
 t=W/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(W/'status.json')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
try:
 assert not json.loads((R/'stage3_supermimic29_20261103/native_universal_20261104/all_four_reference_gate.json').read_text())['training_allowed']
 save();results=[]
 for task in ['pour','bimanual_box_lift','box_push_pull']:
  output=W/task;assert not output.exists();ids=prep['tasks'][task]['candidate_ids']
  source=B/'src/stage3_sample_supported29_batch.py';command=['nice','-n','10',P,'-u',str(source),'--task',task,'--output-dir',str(output),'--count','2','--start-index',str(ids[0]),'--batch-size','2']
  assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2+100*1024**2
  log=(W/(task+'.log')).open('x');child=subprocess.Popen(command,cwd=cfg['trajgen'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state.update(status='sampling_task_pilot',task=task,child_pid=child.pid,command=command);save()
  (W/(task+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,'command':command,'source_sha256':sha(source),'conditions_sha256':sha(B/'src/stage3_supported_task_conditions.py')},indent=2))
  while child.poll() is None:
   own=resources(child.pid);all_owned=snapshot();state.update(resources=own,aggregate_resources=all_owned,elapsed_seconds=time.monotonic()-started);save()
   assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3
   assert all_owned['owned_gpu_bytes']<=24*1024**3 and all_owned['owned_rss_bytes']<=64*1024**3
   assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2 and time.monotonic()-started<5400
   time.sleep(3)
  code=child.wait();log.close();assert code==0,'pilot failed; preserve complete outputs and inspect log'
  report=json.loads((output/'report.json').read_text());assert report['generated_candidates']==2 and report['accepted_references']==0
  results.append({'task':task,'path':str(output/'report.json'),'sha256':sha(output/'report.json')});state.update(generated_candidates=2*len(results),results=results);save()
 state.update(status='three_task_pilot_candidates_generated_postoptimization_pending',generated_candidates=6,accepted_references=0);save()
 (W/'report.json').open('x').write(json.dumps(state,indent=2))
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child and child.poll() is None:stop_group(child.pid)
 raise
