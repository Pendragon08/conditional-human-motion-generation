"""Single supervised bounded FK job for original held-out dance references."""
import argparse,fcntl,hashlib,json,os,pathlib,shutil,subprocess,sys
if sys.flags.optimize or not sys.flags.isolated:raise RuntimeError('isolated unchecked mode required')
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN='user_mixed5000_dance_fk_library_v1'
a=argparse.ArgumentParser();a.add_argument('--request',required=True);a.add_argument('--trusted-request-sha256',required=True);args=a.parse_args()
p=pathlib.Path(args.request)
if os.getuid()!=1013 or p!=ROOT/'receipts'/(RUN+'_request.json') or p.resolve()!=p or p.stat().st_uid!=1013:raise ValueError('request identity')
raw=p.read_bytes()
if hashlib.sha256(raw).hexdigest()!=args.trusted_request_sha256:raise ValueError('request bytes')
s=json.loads(raw)
if s['run']!=RUN or s['role']!='library' or s['fk_authorized'] is not True:raise ValueError('request scope')
for name,h in s['bound_sources'].items():
 q=ROOT/name
 if q.parent!=ROOT/'scripts' or q.resolve()!=q or q.stat().st_uid!=1013 or hashlib.sha256(q.read_bytes()).hexdigest()!=h:raise ValueError('source mismatch '+name)
cache=ROOT/'receipts/mixed5000_dance_no_bytecode_cache_v1'
if cache.exists() or cache.is_symlink():raise ValueError('cache exists')
sys.pycache_prefix=str(cache);sys.dont_write_bytecode=True;sys.path.append(str(ROOT/'scripts'))
from step4_train_only_protocol import RESERVE,LOG,owned,sha,read,atomic
from contact_filter_resources import sample_resources
from contact_process_supervision import run_supervised
from ppo5000_dance_reference_inputs_v1 import accepted_metadata
if sha(__file__)!=s['supervisor_sha256'] or sha(ROOT/'scripts/export_mixed5000_dance_fk_v1.py')!=s['executor_sha256']:raise ValueError('entry identity')
parent=owned(ROOT/'step34_20261029');out=parent/RUN;log=parent/(RUN+'_supervision')
if out.exists() or log.exists():raise ValueError('existing job')
with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 verify='user_augmentation2526_mixed5000_cpu_verify_v1';export='user_augmentation2526_mixed5000_cpu_export_v1';rec=ROOT/'receipts'
 paths=dict(request=rec/(verify+'_request.json'),driver=rec/(verify+'_driver.json'),outer=rec/(verify+'_outer_exit.json'),receipt=parent/verify/'receipt.json',exit=parent/(verify+'_supervision')/'exit.json',process=parent/(verify+'_supervision')/'process_exit.json',resources=parent/(verify+'_supervision')/'resources.jsonl',actor=parent/export/'actor_RMS.private.pth')
 if set(paths)!=set(s['verify_hashes']):raise ValueError('missing policy provenance')
 for k,v in paths.items():
  if sha(owned(v))!=s['verify_hashes'][k]:raise ValueError('policy provenance changed')
 if s['verify_hashes']['actor']!='f2ee798dd0ae3a9b7878c720742d444aa09747ea39829ed96a173f7e2de829c7':raise ValueError('wrong policy')
 m=accepted_metadata(s['validation_scope_binding']);binding={k:m[k] for k in ['train_binding','validation_receipt_sha256','validation_closure','scope_spec_sha256','validation_scope_sha256','source_index_sha256']}
 if binding!=s['input_binding'] or m['estimate']['candidate_count']!=16 or m['estimate']['target_pose_count']!=15517:raise ValueError('dance metadata changed')
 if s['output_upper_bytes']!=m['estimate']['npz_upper_bytes']+128*1024**2:raise ValueError('output budget mismatch')
 floor=RESERVE+LOG+s['output_upper_bytes']+512*1024**2
 if shutil.disk_usage(parent).free<floor:raise ValueError('disk reserve')
 sample_resources(ROOT,None,16*1024**3,floor)
 gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
 if gpu<20*1024**3:raise ValueError('GPU memory')
 log.mkdir(mode=0o700,exist_ok=False)
 def sample(pid):
  if shutil.disk_usage(parent).free<floor:raise RuntimeError('reserved capacity unavailable')
  x=sample_resources(ROOT,pid,16*1024**3,floor);lp=log/'child.log'
  if lp.exists() and lp.stat().st_size>LOG:raise RuntimeError('log budget')
  return x
 cmd=['taskset','-c','0-3','/usr/bin/env','-u','PYTHONOPTIMIZE','-u','PYTHONPATH','-u','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER=PCI_BUS_ID','PYTHONDONTWRITEBYTECODE=1','PYTHONIOENCODING=utf-8','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','bash',str(ROOT/'scripts/run_tracker.sh'),str(ROOT/'scripts/export_mixed5000_dance_fk_v1.py'),'--request',str(p),'--trusted-request-sha256',args.trusted_request_sha256,'--device','cuda:1','--headless']
 proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,7200,15)
 rp=out/'receipt.json';rec=read(rp) if rp.exists() else {}
 proof.update(run=RUN,source_sha256=sha(__file__),request_sha256=sha(p),receipt_sha256=sha(rp) if rp.exists() else None,resource_sha256=sha(log/'resources.jsonl'),process_exit_sha256=sha(log/'process_exit.json'),status=rec.get('status'))
 atomic(log/'exit.json',proof)
 if not (proof['supervision_passed'] and proof['child_exit_code']==0 and rec.get('status')=='PPO5000_DANCE_VAL_REFERENCE_FK_FINISHED' and rec['output_writers_closed'] and rec['exported_clips']==16 and rec['exported_frames']==15517):raise RuntimeError('FK did not finish; retain all evidence')
