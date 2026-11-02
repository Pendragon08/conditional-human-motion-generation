"""Process isolation across complete motions; one canonical output/backup schema.

Workers retain sequential GMR state within each motion. Parent publishes completed
files only after their worker manifest commits a checksum. Raw arrays are private.
"""
import argparse,collections,fcntl,hashlib,json,os,pathlib,subprocess,sys,time,traceback,shutil
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RAM=pathlib.Path('/dev/shm/ouyanghao_textop_20261025')
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
def atomic(path,value):
 temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def read_lines(path):
 if not path.exists():return []
 raw=path.read_bytes();lines=raw.splitlines(keepends=True)
 return [json.loads(x) for x in lines if x.endswith(b'\n')]
def owned(path):
 resolved=path.resolve();assert resolved!=RAM and RAM in resolved.parents and resolved.stat().st_uid==1013
 return resolved
def prepare(args,parallel_spec):
 cfg=json.loads(pathlib.Path(parallel_spec).read_text());base=json.loads(args.spec.read_text())
 assert os.getuid()==1013 and len(set(cfg['cpu_ids']))==cfg['workers']==4
 assert set(cfg['cpu_ids'])<=os.sched_getaffinity(0)
 assert hashlib.sha256(json.dumps(base,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==cfg['warm_base_canonical_hash']
 assert shutil.disk_usage(RAM).free>cfg['minimum_ram_free_bytes'] and shutil.disk_usage(ROOT).free>cfg['minimum_data_free_bytes']
 available=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 assert available>cfg['minimum_system_available_memory_bytes']
 parent=owned(args.output.parent);assert owned(args.pairs.parent)==parent
 assert not args.limit and not args.witness
 args.output.mkdir(mode=0o700,exist_ok=True);output=owned(args.output)
 assert not (output/'prepared_manifest.private.jsonl').exists() and not list(output.glob('*.private.npz'))
 lock=(parent/(output.name+'.parallel.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 work=parent/(output.name+'.parallel_work');work.mkdir(mode=0o700,exist_ok=False);owned(work)
 rows=json.loads(args.pairs.read_text());keys=[x['canonical_source'] for x in rows];assert len(keys)==len(set(keys)) and len(rows)>=4
 for row in rows:
  source=owned(pathlib.Path(row['raw_stageii_candidates'][0]));assert sha(source)==row['raw_sha256']
 shards=[[] for _ in range(4)];cost=[0]*4
 for row in sorted(rows,key=lambda x:(-x['frames'],x['canonical_source'])):
  i=min(range(4),key=lambda i:(cost[i],i));shards[i].append(row);cost[i]+=row['frames']
 assert sorted(x['canonical_source'] for s in shards for x in s)==sorted(keys)
 os.sched_setaffinity(0,set(cfg['cpu_ids']));children=[];started=time.monotonic();completed={};failures={}
 progress={'status':'RUNNING','spec_sha256':sha(args.spec),'parallel_spec_sha256':sha(parallel_spec),
  'source_sha256':sha(pathlib.Path(__file__).with_name('prepare_paper_data.py')),'orchestrator_sha256':sha(__file__),
  'versions':base['required_versions'],'target_dof':29,'feature_dim':69,'output_fps':50,'total_requested':len(rows),
  'completed_sequences':0,'mirrored_sequences':0,'failed_sequences':0,'physical_filter_passed':False,'ready_for_generator_training':False,
  'cpu_only':True,'parallel_workers':4,'cpu_affinity':cfg['cpu_ids'],'threads_per_worker':cfg['torch_threads_per_worker'],'implementation':'same GMR preparation kernel with disjoint-motion CPU processes'}
 def update():
  progress.update(completed_sequences=len(completed),mirrored_sequences=len(completed),failed_sequences=len(failures),processed=len(completed)+len(failures),
   elapsed_seconds=time.monotonic()-started,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  atomic(output/'progress.json',progress)
 def merge():
  for i,child in enumerate(children):
   folder=work/str(i)/'output'
   for item in read_lines(folder/'prepared_manifest.private.jsonl'):
    key=item['canonical_source']
    if key in completed:continue
    assert key in {x['canonical_source'] for x in shards[i]} and key not in failures
    source=owned(pathlib.Path(item['output_file']));assert source.parent==folder and sha(source)==item['output_sha256']
    target=output/source.name;assert not target.exists() and owned(target.parent)==output
    source.replace(target);item['output_file']=str(target)
    with (output/'prepared_manifest.private.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
    completed[key]=item
   for item in read_lines(folder/'failures.private.jsonl'):
    key=item['source']
    if key in failures:continue
    assert key in {x['canonical_source'] for x in shards[i]} and key not in completed
    item['parallel_worker']=i
    with (output/'failures.private.jsonl').open('a') as stream:stream.write(json.dumps(item)+'\n')
    failures[key]=item
 try:
  for i,rows_i in enumerate(shards):
   slot=work/str(i);slot.mkdir();pairfile=slot/'pairs.private.json';atomic(pairfile,rows_i)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
   command=[base['python'],str(pathlib.Path(__file__)), '--worker','--spec',str(args.spec),'--pairs',str(pairfile),'--output',str(slot/'output')]
   with (slot/'worker.log').open('xb') as log:
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
     preexec_fn=lambda:os.sched_setaffinity(0,set(cfg['cpu_ids'])))
   children.append(child)
  progress['worker_pids']=[x.pid for x in children];update()
  while any(c.poll() is None for c in children):merge();update();time.sleep(cfg['poll_seconds'])
  merge();codes=[c.wait() for c in children]
  progress.update(worker_exit_codes=codes,splits=dict(collections.Counter(x['split'] for x in completed.values())),
   paired_segments=sum(len(x['annotations'])+len(x['mirrored_annotations']) for x in completed.values()))
  assert set(completed)|set(failures)==set(keys),'Worker did not account for every input; retained all evidence'
  assert not failures and all(c==0 for c in codes),'Worker failure; original evidence retained'
  assert len(list(output.glob('*.private.npz')))==len(completed)==len(rows)
  progress['status']='PREPROCESSING_FINISHED_FILTER_PENDING';update()
  print(json.dumps(progress),flush=True)
 except BaseException as error:
  # Never kill healthy sibling workers or erase their outputs. Record failure,
  # then wait for this finite motion batch to settle before returning failure.
  for child in children:child.wait()
  try:merge()
  except BaseException:pass
  progress.update(status='PREPROCESSING_FAILED',error=repr(error),traceback=traceback.format_exc(),worker_exit_codes=[x.returncode for x in children]);update();raise
 return progress
def maybe_dispatch(args):
 activation=ROOT/'configs/gmr-parallel-activation.json'
 if not activation.exists():return False
 gate=json.loads(activation.read_text())
 if str(args.output.resolve()) not in gate.get('allowed_outputs',[]):return False
 assert not args.limit and not args.witness
 started=time.monotonic()
 while gate['mode']=='WAIT_FOR_VALIDATION':
  args.output.mkdir(mode=0o700,exist_ok=True)
  atomic(args.output/'progress.json',{'status':'WAITING_FOR_PARALLEL_VALIDATION','completed_sequences':0,'physical_filter_passed':False,'ready_for_generator_training':False})
  assert time.monotonic()-started<gate['maximum_wait_seconds'],'Parallel validation gate expired; inputs retained'
  time.sleep(5);gate=json.loads(activation.read_text())
 if gate['mode']=='SERIAL_FALLBACK':return False
 assert gate['mode']=='PARALLEL_VALIDATED'
 for name,digest in gate['validated_source_hashes'].items():assert sha(ROOT/name)==digest
 for path in gate['validation_receipts']:
  proof=json.loads(pathlib.Path(path).read_text());assert proof['status']=='TEXTOP_PARALLEL_GMR_WITNESS_PASSED'
 prepare(args,ROOT/'configs/gmr-parallel-spec.json');return True
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--spec',type=pathlib.Path,required=True)
 p.add_argument('--pairs',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--parallel-spec',type=pathlib.Path)
 args=p.parse_args();args.limit=None;args.witness=False
 if args.worker:
  assert os.sched_getaffinity(0)=={0,1,2,3} and os.environ['CUDA_VISIBLE_DEVICES']==''
  from prepare_paper_data import main
  main(args,cpu_threads=4)
 else:prepare(args,args.parallel_spec)
