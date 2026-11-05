"""Only generated six-task-pilot candidates; shared GPU0 PyRoki warm reuse."""
import ast,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';W=B/'push_postopt_v7'
sys.path[:0]=[str(S/'src'),str(R/'stage3_supermimic29_20261103/src')]
from stage3_pick500_replacement_queue import snapshot
from stage3_pick_parallel_controller import resources,stop_group
status=B/'push_postopt_v7_guard.json';assert not W.exists() and not status.exists()
audit=B/'pilot_v2/candidate_audit.json';a=json.loads(audit.read_text());assert a['status']=='passed_candidate_file_noise_and_task_binding_audit' and a['candidate_count']==6
previous=[json.loads(x) for x in (B/'pilot_postopt_v2/records.jsonl').read_text().splitlines()]
assert json.loads((B/'pilot_postopt_v2/independent_state_audit.json').read_text())['status']=='passed_independent_rejection_and_state_audit'
rejected={r['candidate_id'] for r in previous if r['status']=='generated_candidate_rejected' and r['candidate_id'] in {102001}}
assert rejected=={102001}
a['records']=[r for r in a['records'] if r['candidate_id'] in rejected]
input_dir=B/'push_postopt_input_v7';input_dir.mkdir(exist_ok=False)
with (input_dir/'records.jsonl').open('x') as f:
 for row in a['records']:f.write(json.dumps(row)+'\n')
(input_dir/'report.json').open('x').write(json.dumps({'status':'sampling_completed_not_postoptimized','generated_candidates':1,'accepted_references':0,'independent_audit_sha256':hashlib.sha256(audit.read_bytes()).hexdigest()},indent=2))
tree=ast.parse((S/'src/stage3_postopt_queue.py').read_text());assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='env' for n in x.targets));ns={'R':R,'S':S,'Path':pathlib.Path,'os':os};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'reuse_existing_PyRoki_environment_only','exec'),ns)
env=ns['env'];env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONDONTWRITEBYTECODE']='1'
source=B/'src/stage3_push_task_postopt_v7.py';ast.parse(source.read_text())
command=['nice','-n','10','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python','-u',str(source),'--records',str(input_dir/'records.jsonl'),'--output-dir',str(W),'--max-candidates','1']
state={'status':'preflight','pid':os.getpid(),'command':command,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'admitted_references':0};child=None;started=time.monotonic()
def save():
 t=status.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2));t.replace(status)
try:
 assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2+100*1024**2
 log=(B/'push_postopt_v7.log').open('x');child=subprocess.Popen(command,cwd=str(B),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(status='generated_pilot_postoptimization_running',child_pid=child.pid);save()
 while child.poll() is None:
  own=resources(child.pid);total=snapshot();state.update(resources=own,aggregate_resources=total,elapsed_seconds=time.monotonic()-started);save()
  assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3
  assert total['owned_gpu_bytes']<=24*1024**3 and total['owned_rss_bytes']<=64*1024**3
  assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2 and time.monotonic()-started<1800
  time.sleep(3)
 assert child.wait()==0,'postoptimizer failed; preserve prefix/logs'
 j=json.loads((W/'report.json').read_text());assert j['status']=='postopt_completed_native_pending' and j['completed']==1
 state.update(status='generated_pilot_postoptimization_completed_native_scene_pending',result=j);save()
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child and child.poll() is None:stop_group(child.pid)
 raise
