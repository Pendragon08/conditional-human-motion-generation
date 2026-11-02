"""Bounded, locked supervisor for exactly one SnapMoGen retarget witness."""
import argparse,base64,fcntl,hashlib,json,os,pathlib,shutil,sys
if sys.flags.optimize or not sys.flags.isolated:raise RuntimeError('isolated interpreter required')
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args()
request=pathlib.Path(a.request)
if os.getuid()!=1013 or request.resolve()!=request or request.stat().st_uid!=1013:raise ValueError('request identity')
raw=request.read_bytes()
if hashlib.sha256(raw).hexdigest()!=a.trusted_request_sha256:raise ValueError('request hash')
j=json.loads(raw);role=j['role'];run=j['run']
if role not in ('author','document') or run!='user_snapmogen_retarget_single_'+role+'_v1' or request!=ROOT/'receipts'/(run+'_request.json'):raise ValueError('request scope')
summary_bytes=base64.b64decode(j['fresh_summary_base64'])
if hashlib.sha256(summary_bytes).hexdigest()!=j['fresh_summary_sha256']:raise ValueError('fresh summary hash')
summary=json.loads(summary_bytes)
if summary['status']!='PASSED' or not all(summary['checks'].values()) or summary['guide']['sha256']!=j['guide_sha256'] or summary['author_result']['sha256']!=j['local_env_author_sha256'] or summary['document_result']['sha256']!=j['local_env_document_sha256']:raise ValueError('fresh summary binding')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def check_meshes():
 meshroot=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR/assets/unitree_g1/meshes')
 if meshroot.resolve()!=meshroot or meshroot.stat().st_uid!=1013 or len(j['meshes'])!=35:raise ValueError('mesh root')
 for x in j['meshes']:
  q=meshroot/x['name'];s=q.stat()
  if q.resolve()!=q or q.is_symlink() or s.st_uid!=1013 or s.st_size!=x['bytes']:raise ValueError('mesh identity')
  h=hashlib.sha1(('blob %d\0'%s.st_size).encode())
  with q.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  a=q.stat()
  if h.hexdigest()!=x['expected_blob'] or (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns):raise ValueError('mesh changed')
for name,digest in j['bound_sources'].items():
 q=ROOT/'scripts'/name
 if q.resolve()!=q or q.is_symlink() or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('source changed: '+name)
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'scripts'))
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
parent=ROOT/'step34_20261029';out=parent/run;log=parent/(run+'_supervision')
if parent.resolve()!=parent or parent.stat().st_uid!=1013 or out.exists() or log.exists():raise ValueError('output identity')
floor=2*1024**3+128*1024**2+64*1024**2+128*1024**2
with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=parent/'user_snapmogen_python311_runtime_v1';inputdir=parent/'user_snapmogen_single_source_v1'
 checks={'input_receipt':inputdir/'input_receipt.json','source_manifest':env/'env_v1/meta/source_manifest.json',
         'env_install':env/'dependencies_install_result.json','env_author':env/'seeded_witness_author_v4/result.json',
         'env_document':env/'seeded_witness_document_v4/result.json'}
 for key,path in checks.items():
  if path.resolve()!=path or path.stat().st_uid!=1013 or sha(path)!=j['binding_hashes'][key]:raise ValueError('input provenance: '+key)
 source=inputdir/'m_st3_00000.bvh'
 if source.is_symlink() or source.stat().st_uid!=1013 or source.stat().st_size!=307721 or sha(source)!=j['source_sha256']:raise ValueError('source bytes')
 py=(env/'install/python/bin/python3.11').resolve()
 if py.stat().st_uid!=1013 or sha(py)!=j['python_binary_sha256']:raise ValueError('interpreter changed')
 if role=='document':
  first=parent/'user_snapmogen_retarget_single_author_v1/receipt.json'
  if first.resolve()!=first or sha(first)!=j['author_receipt_sha256']:raise ValueError('author provenance')
 check_meshes()
 if shutil.disk_usage(parent).free<floor:raise RuntimeError('disk budget')
 sample_resources(ROOT,None,16*1024**3,floor)
 out.mkdir(mode=0o700);log.mkdir(mode=0o700)
 def sample(pid):
  x=sample_resources(ROOT,pid,16*1024**3,floor)
  if x['owned_group_gpu_memory']:raise RuntimeError('CPU-only job acquired GPU memory')
  lp=log/'child.log'
  if lp.exists() and lp.stat().st_size>128*1024**2:raise RuntimeError('log budget')
  return x
 py=env/'install/python/bin/python3.11';worker=ROOT/'scripts/snapmogen_retarget_single_v1.py'
 cmd=['/usr/bin/env','-u','PYTHONOPTIMIZE','-u','PYTHONPATH','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONIOENCODING=utf-8','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',str(py),'-I','-B',str(worker),'--request',str(request),'--trusted-request-sha256',a.trusted_request_sha256]
 proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,3600,15)
 check_meshes()
 receipt=out/'receipt.json';r=json.loads(receipt.read_text()) if receipt.exists() else {}
 proof.update(run=run,request_sha256=sha(request),receipt_sha256=sha(receipt) if receipt.exists() else None,
              resources_sha256=sha(log/'resources.jsonl'),process_exit_sha256=sha(log/'process_exit.json'),status=r.get('status'))
 with (log/'exit.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 if not(proof['supervision_passed'] and proof['child_exit_code']==0 and r.get('status')=='SNAPMOGEN_ONE_SOURCE_RETARGETED_AWAITING_INDEPENDENT_AUDIT' and r['retargeted_frames']==406 and r['shape_optimizer_steps']==3000):raise RuntimeError('Witness did not finish; preserve evidence')
