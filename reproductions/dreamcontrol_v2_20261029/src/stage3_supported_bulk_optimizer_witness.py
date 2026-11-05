"""Consume two completed production rows once; retain them for the full cohort."""
import ast,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103';W=B/'bulk_v1/optimization_witness';producer=B/'bulk_v1/cohorts/pour_0001/candidates'
sys.path[:0]=[str(S/'src'),str(R/'stage3_supermimic29_20261103/src')]
from stage3_pick500_replacement_queue import snapshot
from stage3_pick_parallel_controller import resources,stop_group
W.mkdir(exist_ok=False);state={'status':'waiting_two_completed_production_rows','pid':os.getpid(),'formal_training_started':False};child=None;began=time.monotonic()
def save():
 t=W/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(W/'status.json')
try:
 save()
 while True:
  p=producer/'records.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines(keepends=True) if x.endswith('\n')] if p.exists() else []
  if len(rows)>=2:break
  assert time.monotonic()-began<300;time.sleep(3)
 rows=rows[:2];assert [r['candidate_id'] for r in rows]==[11000000,11000001]
 for r in rows:assert hashlib.sha256(pathlib.Path(r['path']).read_bytes()).hexdigest()==r['sha256']
 inputs=W/'inputs';inputs.mkdir()
 (inputs/'records.jsonl').open('x').write(''.join(json.dumps(r)+'\n' for r in rows));(inputs/'report.json').open('x').write(json.dumps({'status':'sampling_completed_not_postoptimized','generated_candidates':2,'scope':'two fully completed immutable rows; parent32-candidate producer continues'}))
 tree=ast.parse((S/'src/stage3_postopt_queue.py').read_text());assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='env' for n in x.targets));ns={'R':R,'S':S,'Path':pathlib.Path,'os':os};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'reuse existing environment','exec'),ns)
 env=ns['env'];env['CUDA_VISIBLE_DEVICES']='0';env['PYTHONDONTWRITEBYTECODE']='1'
 source=B/'src/stage3_supported_bulk_postopt_v1.py';ast.parse(source.read_text());command=['nice','-n','10','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python','-u',str(source),'--records',str(inputs/'records.jsonl'),'--output-dir',str(W/'postopt'),'--max-candidates','2']
 log=(W/'postopt.log').open('x');child=subprocess.Popen(command,cwd=str(B),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(status='two_stage_production_optimizer_running',child_pid=child.pid,command=command,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest());save();run_start=time.monotonic()
 while child.poll() is None:
  own=resources(child.pid);total=snapshot();state.update(resources=own,aggregate_resources=total,elapsed_seconds=time.monotonic()-began);save()
  assert own['gpu_bytes']<=12*1024**3 and own['rss_bytes']<=32*1024**3 and total['owned_gpu_bytes']<=24*1024**3 and total['owned_rss_bytes']<=64*1024**3
  assert shutil.disk_usage('/data').free>=8*1024**3+456*1024**2 and time.monotonic()-run_start<1800;time.sleep(3)
 assert child.wait()==0;report=json.loads((W/'postopt/report.json').read_text());assert report['completed']==2
 state.update(status='production_optimizer_first_two_completed_reuse_required',result=report);save()
except BaseException as e:
 state.update(status='failed_preserved',error=repr(e),traceback=traceback.format_exc());save()
 if child and child.poll() is None:stop_group(child.pid)
 raise
