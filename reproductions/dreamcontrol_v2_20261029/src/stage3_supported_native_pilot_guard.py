"""Sequential task scenes for existing independently checked robot pilot states."""
import ast,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';W=B/'native_pilot_v1'
sys.path[:0]=[str(S/'src'),str(R/'stage3_supermimic29_20261103/src')]
from stage3_pick500_replacement_queue import snapshot
from stage3_pick_parallel_controller import resources,stop_group
assert not W.exists();W.mkdir();state={'status':'initializing','pid':os.getpid(),'formal_training_started':False};child=None;began=time.monotonic()
def save():
 t=W/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(W/'status.json')
try:
 staged=json.loads((B/'staged_kinematic_pilots.json').read_text());assert len(staged['staged_candidates'])==4
 spec=json.loads((R/'stage3_supermimic29_20261103/runtime_bridge_spec.json').read_text())
 source=B/'src/stage3_supported_native_replay.py';ast.parse(source.read_text());env={**os.environ,**spec['env']};env['PYTHONPATH']=str(B/'src')+':'+env['PYTHONPATH'];env.pop('CUDA_VISIBLE_DEVICES',None)
 results=[];save()
 for task in ['pour','bimanual_box_lift']:
  selected=[r for r in staged['staged_candidates'] if r['task']==task];assert len(selected)==2
  output=W/task;command=[spec['python'],'-u',str(source),'--headless','--device','cuda:0','--task',task,'--output-dir',str(output)]
  for r in selected:
   assert r['passed_requested_kinematic_checks'] and hashlib.sha256(pathlib.Path(r['output']).read_bytes()).hexdigest()==r['output_sha256']
   command.extend(['--candidate-dir',str(pathlib.Path(r['output']).parent)])
  assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2+100*1024**2
  log=(W/(task+'.log')).open('x');child=subprocess.Popen(command,cwd=spec['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state.update(status='native_task_scene_replay_running',task=task,child_pid=child.pid,command=command);save()
  (W/(task+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,'command':command,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2));finished=None;phase=time.monotonic()
  while child.poll() is None:
   own=resources(child.pid);total=snapshot();state.update(resources=own,aggregate_resources=total,elapsed_seconds=time.monotonic()-began);save()
   assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3
   assert total['owned_gpu_bytes']<=24*1024**3 and total['owned_rss_bytes']<=64*1024**3
   assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2 and time.monotonic()-phase<600
   if (output/'report.json').exists():
    j=json.loads((output/'report.json').read_text())
    if j['status']=='native_task_replay_completed':
     if finished is None:finished=time.monotonic()
     if time.monotonic()-finished>30:stop_group(child.pid);state['cleanup_group_release_after_completed_report']=True;break
   time.sleep(2)
  log.close();assert (output/'report.json').exists(),'native process ended without a completed report; inspect preserved log'
  j=json.loads((output/'report.json').read_text());assert j['status']=='native_task_replay_completed' and j['candidates']==2 and j['frames']==180
  results.append({'task':task,'report':str(output/'report.json'),'sha256':hashlib.sha256((output/'report.json').read_bytes()).hexdigest(),'admitted':j['admitted_references']});state.update(results=results);save()
 state.update(status='native_four_pilot_scene_checks_completed',admitted_references=sum(r['admitted'] for r in results));save();(W/'report.json').open('x').write(json.dumps(state,indent=2))
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child and child.poll() is None:stop_group(child.pid)
 raise
