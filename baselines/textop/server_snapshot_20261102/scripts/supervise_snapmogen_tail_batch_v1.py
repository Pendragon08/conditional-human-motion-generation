"""Resource and identity supervisor for full SnapMoGen retargeting."""
import argparse,base64,fcntl,hashlib,json,os,pathlib,shutil,sys
if sys.flags.optimize or not sys.flags.isolated:raise RuntimeError('isolated interpreter required')
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');RUNS={'user_snapmogen_tail_batch_author_v1','user_snapmogen_tail_batch_document_v1','user_snapmogen_tail_batch_v1'}
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args();request=pathlib.Path(a.request)
if os.getuid()!=1013 or request.resolve()!=request or sha(request)!=a.trusted_request_sha256:raise ValueError('request identity')
j=json.loads(request.read_text());RUN=j['run']
if RUN not in RUNS or request!=ROOT/'receipts'/(RUN+'_request.json') or j['authorized'] is not True:raise ValueError('scope')
summary=base64.b64decode(j['fresh_summary_base64'])
if hashlib.sha256(summary).hexdigest()!=j['fresh_summary_sha256'] or json.loads(summary)['status']!='PASSED':raise ValueError('fresh witness')
for name,digest in j['bound_sources'].items():
 q=ROOT/'scripts'/name
 if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('bound source '+name)
sys.path.insert(0,str(ROOT/'scripts'))
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
parent=ROOT/'step34_20261029';out=parent/RUN;log=parent/(RUN+'_supervision');floor=12884901888+1747940638+416*1024**2
with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if out.exists() or log.exists() or shutil.disk_usage(parent).free<floor:raise ValueError('existing output or capacity')
 for name,digest in j['provenance_hashes'].items():
  q=ROOT/name
  if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('provenance '+name)
 env=parent/'user_snapmogen_python311_runtime_v1';py=env/'install/python/bin/python3.11'
 if sha(py.resolve())!=j['python_binary_sha256']:raise ValueError('Python binary')
 inp=parent/'user_snapmogen_full_sources_v1';transfer=json.loads((inp/'receipt.json').read_text())
 if transfer['status']!='SNAPMOGEN_TEST_SOURCE_BYTES_COPIED_NOT_RETARGETED' or transfer['sources']!=1120 or transfer['split_sha256']!=j['split_sha256'] or sha(inp/'test_ids.txt')!=j['split_sha256']:raise ValueError('input closure')
 if len(j['meshes'])!=35 or len(j['sources'])!=(1116 if RUN=='user_snapmogen_tail_batch_v1' else 2):raise ValueError('manifest counts')
 meshroot=pathlib.Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/GMR/assets/unitree_g1/meshes')
 for x in j['meshes']:
  q=meshroot/x['name'];st=q.stat()
  if q.resolve()!=q or q.stat().st_uid!=1013 or st.st_size!=x['bytes']:raise ValueError('mesh identity')
  h=hashlib.sha1(('blob %d\0'%st.st_size).encode())
  with q.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
  if h.hexdigest()!=x['expected_blob']:raise ValueError('mesh hash')
 sample_resources(ROOT,None,16*1024**3,floor);out.mkdir(mode=0o700);log.mkdir(mode=0o700)
 def sample(pid):
  row=sample_resources(ROOT,pid,16*1024**3,floor)
  if row['owned_group_gpu_memory']:raise RuntimeError('unexpected GPU memory')
  if (log/'child.log').exists() and (log/'child.log').stat().st_size>128*1024**2:raise RuntimeError('log budget')
  return row
 cmd=['/usr/bin/env','-u','PYTHONOPTIMIZE','-u','PYTHONPATH','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONIOENCODING=utf-8','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',str(py),'-I','-B',str(ROOT/'scripts/snapmogen_tail_batch_v1.py'),'--request',str(request),'--trusted-request-sha256',a.trusted_request_sha256]
 proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,72*3600,15)
 receipt=out/'receipt.json';r=json.loads(receipt.read_text()) if receipt.exists() else {}
 proof.update(run=RUN,request_sha256=sha(request),receipt_sha256=sha(receipt) if receipt.exists() else None,resources_sha256=sha(log/'resources.jsonl'),process_exit_sha256=sha(log/'process_exit.json'),status=r.get('status'))
 with (log/'exit.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 if not(proof['supervision_passed'] and r.get('status')=='SNAPMOGEN_TAIL_BATCH_FINISHED_AWAITING_AUDIT' and r['retargeted_or_reused']+r['failed_sources']==len(j['sources']) and r['output_writers_closed']):raise RuntimeError('retarget did not close; preserve evidence')
