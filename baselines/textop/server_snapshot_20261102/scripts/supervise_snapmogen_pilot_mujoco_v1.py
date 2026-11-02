"""Run one authenticated SnapMoGen policy pilot under the existing common lock."""
import argparse,fcntl,hashlib,json,os,pathlib,shutil,sys
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');RUN='user_snapmogen_pilot_mujoco_v1'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args()
request=pathlib.Path(a.request)
if not __debug__ or os.getuid()!=1013 or request.resolve()!=request or request!=ROOT/'receipts'/(RUN+'_request.json') or sha(request)!=a.trusted_request_sha256:raise ValueError('request')
j=json.loads(request.read_text())
if j['run']!=RUN or j['authorized'] is not True:raise ValueError('scope')
for name,digest in j['bound_sources'].items():
 q=ROOT/name
 if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('source: '+name)
sys.path.insert(0,str(ROOT/'scripts'))
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised
from augmentation2526_mixed5000_heldout_inputs_v1 import accept
parent=ROOT/'step34_20261029';out=parent/RUN;log=parent/(RUN+'_supervision');floor=2*1024**3+128*1024**2+64*1024**2+128*1024**2
os.sched_setaffinity(0,{0,1,2,3})
with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if out.exists() or log.exists() or shutil.disk_usage(parent).free<floor:raise ValueError('existing output or disk budget')
 if accept(j['input_binding'])['actor_sha256']!=j['actor_sha256']:raise ValueError('actor binding')
 if sha(parent/'user_mixed5000_onnx_export_v1/receipt.json')!=j['onnx_receipt_sha256'] or sha(parent/'user_mixed5000_onnx_export_v1/policy.onnx')!=j['onnx_sha256']:raise ValueError('ONNX provenance')
 for name,digest in j['provenance_hashes'].items():
  q=ROOT/name
  if q.resolve()!=q or q.stat().st_uid!=1013 or sha(q)!=digest:raise ValueError('provenance: '+name)
 if sha(parent/'user_snapmogen_retarget_single_author_v1/reference.private.npz')!=j['retarget_reference_sha256']:raise ValueError('reference')
 sample_resources(ROOT,None,8*1024**3,floor)
 out.mkdir(mode=0o700);log.mkdir(mode=0o700)
 def sample(pid):
  row=sample_resources(ROOT,pid,8*1024**3,floor)
  if row['owned_group_gpu_memory']:raise RuntimeError('unexpected GPU memory')
  if (log/'child.log').exists() and (log/'child.log').stat().st_size>128*1024**2:raise RuntimeError('log budget')
  return row
 py='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'
 cmd=['/usr/bin/env','-u','PYTHONPATH','-u','PYTHONOPTIMIZE','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4',py,'-I','-B',str(ROOT/'scripts/snapmogen_pilot_mujoco_v1.py'),'--request',str(request),'--trusted-request-sha256',a.trusted_request_sha256]
 proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,1800,15)
 recpath=out/'receipt.json';rec=json.loads(recpath.read_text()) if recpath.exists() else {}
 proof.update(run=RUN,request_sha256=sha(request),receipt_sha256=sha(recpath) if recpath.exists() else None,resource_sha256=sha(log/'resources.jsonl'),process_exit_sha256=sha(log/'process_exit.json'))
 with (log/'exit.json').open('x') as f:json.dump(proof,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 if not(proof['supervision_passed'] and rec.get('status')=='SNAPMOGEN_PILOT_MUJOCO_EVALUATED_AWAITING_AUDIT' and rec.get('output_writers_closed') and rec.get('optimizer_updates')==0):raise RuntimeError('pilot failed; retain evidence')
