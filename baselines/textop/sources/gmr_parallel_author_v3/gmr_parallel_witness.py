"""Compare serial4threads and four independent4thread workers sharing CPUs0-3 on full motions."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time,traceback
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');RAM=pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert a.run.replace('_','').isalnum() and os.getuid()==1013
out=ROOT/'parallel_validation'/a.run;out.mkdir(parents=True,exist_ok=False)
private=RAM/'parallel_validation'/a.run;private.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','physical_filter_executed':False,'dataset_accepted':False,'full_generator_training_started':False}
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return h.hexdigest()
def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2))
try:
 cfg=json.loads((ROOT/'configs/gmr-parallel-spec.json').read_text());base=json.loads((ROOT/'configs/data-prep-env-spec.json').read_text())
 rows=json.loads((RAM/'cmu1330_v1/pairs.private.json').read_text())
 records=[json.loads(x) for x in (RAM/'cmu1330_v1/prepared1330_v1/prepared_manifest.private.jsonl').read_text().splitlines()]
 records=sorted(records,key=lambda x:(x['frames50'],x['canonical_source']))
 chosen=[records[int(q*(len(records)-1))] for q in cfg['diagnostic_quantiles']];keys={x['canonical_source'] for x in chosen}
 assert len(keys)==cfg['diagnostic_sources']==8
 pairfile=private/'pairs.private.json';pairfile.write_text(json.dumps([x for x in rows if x['canonical_source'] in keys]))
 receipt.update(source_hashes={name:sha(ROOT/name) for name in ['scripts/prepare_paper_data.py','scripts/parallel_prepare_data.py','scripts/gmr_parallel_witness.py','configs/gmr-parallel-spec.json','configs/data-prep-env-spec.json']},
  diagnostic_source_count=8,total_output_pose_frames=sum(x['frames50'] for x in chosen),status='RUNNING_SERIAL_THEN_PARALLEL',runs={})
 save()
 for name,script,extra in [('serial','prepare_paper_data.py',[]),('parallel','parallel_prepare_data.py',['--parallel-spec',str(ROOT/'configs/gmr-parallel-spec.json')])]:
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
  command=[base['python'],str(ROOT/'scripts'/script),'--spec',str(ROOT/'configs/data-prep-env-spec.json'),'--pairs',str(pairfile),'--output',str(private/name)]+extra
  started=time.monotonic();samples=[]
  with (out/(name+'.log')).open('xb') as log:
   child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,set(cfg['cpu_ids'])))
   while child.poll() is None:
    if time.monotonic()-started>cfg['timeout_seconds']:raise TimeoutError('Finite diagnostic exceeded timeout; children and evidence retained for inspection')
    own=[]
    for proc in pathlib.Path('/proc').iterdir():
     if not proc.name.isdigit():continue
     try:
      cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
      if proc.stat().st_uid!=1013 or str(private) not in cmd:continue
      own.append({'pid':int(proc.name),'cpu_affinity':sorted(os.sched_getaffinity(int(proc.name))),
       'rss_kib':next((int(x.split()[1]) for x in (proc/'status').read_text().splitlines() if x.startswith('VmRSS:')),0)})
     except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    samples.append({'seconds':time.monotonic()-started,'processes':own});time.sleep(1)
   code=child.wait()
  elapsed=time.monotonic()-started;progress=json.loads((private/name/'progress.json').read_text())
  receipt['runs'][name]={'child_exit_code':code,'elapsed_seconds':elapsed,'source_count':progress['completed_sequences'],'failed_sequences':progress['failed_sequences'],
   'status':progress['status'],'worker_pids':progress.get('worker_pids',[]),'worker_exit_codes':progress.get('worker_exit_codes',[]),'resource_samples':samples}
  save();assert code==0 and progress['completed_sequences']==8 and progress['failed_sequences']==0 and progress['status']=='PREPROCESSING_FINISHED_FILTER_PENDING'
 import numpy as np
 def manifest(folder):return {x['canonical_source']:x for x in (json.loads(s) for s in (folder/'prepared_manifest.private.jsonl').read_text().splitlines())}
 serial=manifest(private/'serial');parallel=manifest(private/'parallel');assert serial.keys()==parallel.keys()==keys
 maxima={};array_checks=0;annotation_checks=0;original_maxima={}
 originals={x['canonical_source']:x for x in chosen}
 for key in sorted(keys):
  for label,other in [('parallel',parallel[key]),('original',originals[key])]:
   left=serial[key]
   for field in ['split','raw_sha256','frames50','dof','features','annotations','mirrored_annotations','rejected_annotations','joint_names']:
    assert left[field]==other[field],(label,field);annotation_checks+=1
   lp=pathlib.Path(left['output_file']);rp=pathlib.Path(other['output_file'])
   assert sha(lp)==left['output_sha256'] and sha(rp)==other['output_sha256']
   with np.load(lp,allow_pickle=False) as l,np.load(rp,allow_pickle=False) as r:
    assert set(l.files)==set(r.files)
    for name in l.files:
     assert l[name].shape==r[name].shape and l[name].dtype==r[name].dtype and np.isfinite(l[name]).all() and np.isfinite(r[name]).all()
     error=float(np.max(np.abs(l[name]-r[name])));target=maxima if label=='parallel' else original_maxima;target[name]=max(target.get(name,0.),error)
     if 'contact' in name or name=='times_s':assert np.array_equal(l[name],r[name]),(label,name,error)
     else:assert np.allclose(l[name],r[name],atol=cfg['comparison']['float_atol'],rtol=cfg['comparison']['float_rtol']),(label,name,error)
     array_checks+=1
 assert receipt['runs']['parallel']['worker_exit_codes']==[0,0,0,0]
 samples=receipt['runs']['parallel']['resource_samples'];pids=set(receipt['runs']['parallel']['worker_pids'])
 observed={p['pid'] for s in samples for p in s['processes'] if p['pid'] in pids and p['cpu_affinity']==cfg['cpu_ids']}
 assert observed==pids and all(set(p['cpu_affinity'])<=set(cfg['cpu_ids']) for s in samples for p in s['processes'])
 receipt.update(status='TEXTOP_PARALLEL_GMR_WITNESS_PASSED',array_comparisons=array_checks,metadata_field_comparisons=annotation_checks,
  parallel_vs_serial_max_abs_errors=maxima,new_serial_vs_previous_kernel_max_abs_errors=original_maxima,
  contacts_exact=True,timestamps_exact=True,annotations_exact=True,disjoint_sources_verified=True,canonical_top_level_outputs=8,
  shared_four_core_worker_affinity_verified=True,worker_cpu_ids=cfg['cpu_ids'],observed_speedup=receipt['runs']['serial']['elapsed_seconds']/receipt['runs']['parallel']['elapsed_seconds'],
  outputs_private=True,output_writers_closed=True,limitation='Eight full CMU motions only; throughput is a measured sample, not a guaranteed full-cohort speedup or physical acceptance')
 save();print(json.dumps({k:v for k,v in receipt.items() if k!='runs'}),flush=True)
except BaseException as e:receipt.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());save();raise
