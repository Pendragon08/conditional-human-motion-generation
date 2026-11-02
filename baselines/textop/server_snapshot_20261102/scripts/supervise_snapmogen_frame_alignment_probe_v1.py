"""Draft: exclusive resource-bounded supervisor for the frame-index probe."""
import argparse,fcntl,hashlib,json,os,pathlib,shutil,sys
if sys.flags.optimize or not sys.flags.isolated:raise RuntimeError('isolated interpreter required')
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');RUN='user_snapmogen_frame_alignment_probe_v1'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args();request=pathlib.Path(a.request)
if os.getuid()!=1013 or request.resolve()!=request or request!=ROOT/'receipts'/(RUN+'_request.json') or sha(request)!=a.trusted_request_sha256:raise ValueError('request identity')
j=json.loads(request.read_text())
if j['run']!=RUN or j['authorized'] is not True:raise ValueError('scope')
for name,digest in j['bound_sources'].items():
 q=ROOT/'scripts'/name
 if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('source changed '+name)
sys.path.insert(0,str(ROOT/'scripts'))
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
parent=ROOT/'step34_20261029';out=parent/RUN;log=parent/(RUN+'_supervision');inputs=parent/'user_snapmogen_feature_alignment_v1';floor=2*1024**3+128*1024**2+32*1024**2
with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if out.exists() or log.exists() or shutil.disk_usage(parent).free<floor:raise ValueError('existing output or capacity')
 for name,digest in j['provenance_hashes'].items():
  q=ROOT/name
  if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('provenance '+name)
 if sha(inputs/'receipt.json')!=j['feature_receipt_sha256']:raise ValueError('feature receipt')
 for item in j['samples']:
  q=inputs/(item['name']+'.npy')
  if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=item['feature_sha256']:raise ValueError('feature sample')
 for item in j['official_files']:
  q=inputs/item['filename']
  if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=item['sha256']:raise ValueError('official source')
 py=parent/'user_snapmogen_python311_runtime_v1/install/python/bin/python3.11'
 if sha(py.resolve())!=j['python_binary_sha256']:raise ValueError('Python binary')
 sample_resources(ROOT,None,8*1024**3,floor);out.mkdir(mode=0o700);log.mkdir(mode=0o700)
 def sample(pid):
  row=sample_resources(ROOT,pid,8*1024**3,floor)
  if row['owned_group_gpu_memory']:raise RuntimeError('unexpected GPU memory')
  if (log/'child.log').exists() and (log/'child.log').stat().st_size>128*1024**2:raise RuntimeError('log budget')
  return row
 cmd=['/usr/bin/env','-u','PYTHONOPTIMIZE','-u','PYTHONPATH','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONIOENCODING=utf-8','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',str(py),'-I','-B',str(ROOT/'scripts/snapmogen_frame_alignment_probe_v1.py'),'--request',str(request),'--trusted-request-sha256',a.trusted_request_sha256]
 proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,900,15)
 receipt=out/'receipt.json';r=json.loads(receipt.read_text()) if receipt.exists() else {}
 proof.update(run=RUN,request_sha256=sha(request),receipt_sha256=sha(receipt) if receipt.exists() else None,resources_sha256=sha(log/'resources.jsonl'),process_exit_sha256=sha(log/'process_exit.json'),status=r.get('status'))
 with (log/'exit.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 if not(proof['supervision_passed'] and r.get('status')=='SNAPMOGEN_FEATURE_ALIGNMENT_MEASURED_AWAITING_AUDIT' and len(r['samples'])==5 and r['decision'] in ('CONSISTENT_FOR_TESTED_SOURCES','INCONCLUSIVE')):raise RuntimeError('probe did not close; preserve evidence')
