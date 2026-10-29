"""Future CPU audit/export for the real user-selected5000 PPO state only.

No simulator, optimizer step, CUDA RNG restore, or semantic/online acceptance.
This entry cannot run until an authorized request and completed producer exist.
"""
import argparse,fcntl,json,os,pathlib,shutil,time,traceback
from step4_ppo5000_frozen_protocol import ROOT,RESERVE,LOG,owned,read,sha,fingerprint,accepted_stop,accepted_export

def require_owned_supervisor(request):
 pid=os.getppid();found=False
 for _ in range(8):
  if pid<=1:break
  proc=pathlib.Path('/proc')/str(pid)
  try:
   argv=(proc/'cmdline').read_bytes().decode().split('\0')
   if proc.stat().st_uid==1013 and str(ROOT/'scripts/supervise_ppo5000_frozen_cpu.py') in argv and str(request) in argv:
    found=True;break
   pid=int((proc/'stat').read_text().rsplit(')',1)[1].split()[1])
  except (FileNotFoundError,ProcessLookupError):break
 assert found,'CPU model must run under its owned lock-holding supervisor'
 with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  raise AssertionError('CPU supervisor must hold the common model lock')

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
 request=owned(a.request);s=read(request)
 assert os.getuid()==1013 and s['authorized'] is True and s['role'] in ['export','verify']
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))=={0,1,2,3}
 assert sha(__file__)==s['executor_sha256'];require_owned_supervisor(request)
 for name,digest in s['bound_sources'].items():assert sha(owned(ROOT/name))==digest
 train=accepted_stop(s['training_binding'])
 export=accepted_export(s['export_binding']) if s['role']=='verify' else None
 parent=owned(ROOT/'step34_numeric_v1');out=parent/s['run'];out.mkdir(mode=0o700,exist_ok=False)
 started=time.monotonic()
 rec=dict(status='AUDITING_PPO5000_INTERRUPTED_FULLSTATE_ON_CPU',run=s['run'],role=s['role'],
  source_sha256=sha(__file__),request_sha256=sha(request),training_binding=train['binding'],
  output_writers_closed=False,optimizer_updates_performed=0,new_CUDA_contexts=0,CUDA_RNG_restored=False,
  heldout_tracking_accepted=False,online_latency_accepted=False)
 def persist():
  rec['elapsed_seconds']=time.monotonic()-started;tmp=out/'receipt.partial'
  with tmp.open('w') as f:json.dump(rec,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  tmp.replace(out/'receipt.json')
 persist()
 try:
  assert shutil.disk_usage(parent).free>=RESERVE+LOG+(32*1024**2 if s['role']=='export' else 0)
  import torch
  from step4_train_only_ppo_runtime_v4 import content,equal
  from step4_ppo5000_cpu_core import audit_and_payload,independent_reload
  assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
  cp=train['binding']['checkpoint'];before=fingerprint(cp['path'])
  assert before['sha256']==cp['sha256'] and before['bytes']==cp['bytes']
  state=torch.load(owned(cp['path']),map_location='cpu',weights_only=False)
  assert content(state)==cp['content_sha256']
  payload,audit,expected,probes=audit_and_payload(state,train['binding'])
  assert audit['fullstate_metadata']['completed_iterations']==5000
  assert audit['fullstate_metadata']['Adam_updates']==100000
  if s['role']=='export':
   path=out/'actor_RMS.private.pth';partial=out/'actor_RMS.private.pth.partial'
   assert not path.exists() and not partial.exists()
   with partial.open('xb') as f:
    os.chmod(partial,0o600);torch.save(payload,f);f.flush();os.fsync(f.fileno())
   assert 0<partial.stat().st_size<=32*1024**2
   loaded=torch.load(partial,map_location='cpu',weights_only=False);assert equal(loaded,payload)
   assert shutil.disk_usage(parent).free>=RESERVE+LOG
   partial.rename(path);fd=os.open(str(out),os.O_RDONLY)
   try:os.fsync(fd)
   finally:os.close(fd)
  else:
   path=owned(export['binding']['inference']['path'])
   loaded=torch.load(path,map_location='cpu',weights_only=False)
   assert equal(loaded,payload)
  verification=independent_reload(loaded,expected,probes)
  assert verification['independent_actor_RMS_reload_exact'] is True
  assert fingerprint(cp['path'])==before and accepted_stop(train['binding'])['binding']==train['binding']
  fp=fingerprint(path);artifact={k:fp[k] for k in ['path','bytes','sha256']}
  if export:assert artifact==export['binding']['inference']
  rec.update(status='PPO5000_FROZEN_CPU_EXPORTED' if s['role']=='export' else 'PPO5000_FROZEN_FRESH_CPU_VERIFIED',
   output_writers_closed=True,inference=artifact,fullstate_exact_5000_100000=True,
   independent_actor_RMS_reload_exact=True,synthetic_probe_only=True,original_fullstate_unchanged=True,
   fullstate_before_and_after=before,audit=audit,probe=verification,export_binding=s['export_binding'])
  assert not torch.cuda.is_initialized();persist()
 except BaseException as error:
  rec.update(status='FAILED_PPO5000_FROZEN_CPU_OUTPUTS_RETAINED',error=repr(error),traceback=traceback.format_exc())
  persist();raise
if __name__=='__main__':main()
