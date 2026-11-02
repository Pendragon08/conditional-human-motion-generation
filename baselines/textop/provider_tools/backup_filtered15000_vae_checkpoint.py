"""Copy one real, stable filtered VAE checkpoint to private D storage.

The producing trainer already performed its exact state reload. This tool only
verifies provenance and bytes; it never imports Torch, reads tensors, signals a
trainer, dispatches a server job, or acquires the shared GPU lock. Partial and
failed copies are retained and must be inspected before any later attempt.
"""
import argparse, ast, hashlib, json, msvcrt, os, pathlib, re, shutil, sys, time
from backup_filter_recovery_checkpoint import recorded_client, recorded_remote
from filter_recovery_windows_paths import guard_local
from session import SSH

B=pathlib.Path(__file__).resolve().parents[1]
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RUN='filtered15000_vae200k_train_v1'
SPEC_SHA='b7eb6e95b082fd3902099ea718ec07bd42b3418c49c179f88f5438e5c5d2c6fa'
BINDING_SHA='87a2f61683d930bf5fe69a9113713c90137c43318766627975c6d11accc20c1c'
RESERVE=2*1024**3
METADATA_ALLOWANCE=8*1024**2

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def dump_new(path,data):
 with pathlib.Path(path).open('x',encoding='utf-8') as f:
  json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def validate_collected(j,step,spec):
 assert j['kind']=='vae' and j['run']==RUN and j['arrays_read']==j['model_loads']==0
 r=j['files']['receipt']
 assert r['status'] in ['RUNNING_FILTERED15000_VAE_TRAINING','FILTERED15000_VAE_200000_TRAINING_FINISHED']
 assert r['spec_sha256']==SPEC_SHA and r['source_sha256']==spec['entrypoint_sha256']
 assert r['runtime_binding_sha256']==BINDING_SHA
 for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:assert r[k]==spec[k]
 assert r['completed_primitive_attempts']==r['optimizer_updates'] and r['skipped_updates']==0
 choices=[x for x in r.get('checkpoints',[]) if x['manager_step']==step]
 assert len(choices)==1,'Collect a real saved checkpoint before copying; progress is not a checkpoint'
 ck=choices[0]
 assert ck['actual_optimizer_updates']==step and r['completed_primitive_attempts']>=step
 expected=spec['checkpoint_dir']+'/vae_state_'+str(step)+'.pth'
 assert ck['path']==expected and 0<ck['bytes']<=spec['checkpoint_policy']['checkpoint_upper_bound_bytes']
 assert re.fullmatch('[0-9a-f]{64}',ck['sha256'])
 proof=r.get('latest_file_resume_verification',{})
 assert all(proof.get(k) is True for k in ['model_restored_exactly','optimizer_restored_exactly','ema_restored_exactly','rng_and_generator_restored_exactly','clock_restored_exactly','pending_evaluation_fields_restored'])
 if r['status']=='RUNNING_FILTERED15000_VAE_TRAINING':assert r['completed_primitive_attempts']>step,'Wait until the producer completed reload and advanced'
 return ck

def remote_script(step,expected,prior=None):
 code=r'''
import hashlib,json,os,pathlib,resource,stat,time
assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});resource.setrlimit(resource.RLIMIT_CPU,(45,45))
r=pathlib.Path(ROOT_LITERAL);step=STEP_LITERAL;expected=EXPECTED_LITERAL;prior=PRIOR_LITERAL
specpath=r/'configs/filtered15000-vae-spec.json';raw=specpath.read_bytes()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert hashlib.sha256(raw).hexdigest()==SPEC_LITERAL
s=json.loads(raw);assert s['train_authorized'] and s['witness_authorized']
assert sha(r/'scripts/run_filtered15000_vae.py')==s['entrypoint_sha256']
assert sha(r/'scripts/supervise_filtered15000_vae.py')==s['supervisor_sha256']
for name,h in s['bound_sources'].items():assert sha(r/name)==h,name
accept=r/'receipts/filtered15000_data_acceptance_v1.json';assert sha(accept)==s['data_acceptance_sha256']
rp=r/'filtered15000_20261027/vae_runs/filtered15000_vae200k_train_v1/receipt.json'
recraw=rp.read_bytes();rec=json.loads(recraw)
assert rec['mode']=='train' and rec['status'] in ['RUNNING_FILTERED15000_VAE_TRAINING','FILTERED15000_VAE_200000_TRAINING_FINISHED']
assert rec['spec_sha256']==SPEC_LITERAL and rec['source_sha256']==s['entrypoint_sha256'] and rec['runtime_binding_sha256']==BINDING_LITERAL
for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:assert rec[k]==s[k]
assert rec['completed_primitive_attempts']==rec['optimizer_updates'] and rec['skipped_updates']==0
cks=[x for x in rec.get('checkpoints',[]) if x['manager_step']==step];assert len(cks)==1 and cks[0]==expected
proof=rec['latest_file_resume_verification']
assert all(proof.get(k) is True for k in ['model_restored_exactly','optimizer_restored_exactly','ema_restored_exactly','rng_and_generator_restored_exactly','clock_restored_exactly','pending_evaluation_fields_restored'])
assert rec['completed_primitive_attempts']>step or rec['status']=='FILTERED15000_VAE_200000_TRAINING_FINISHED'
src=pathlib.Path(expected['path']);home=pathlib.Path('/home/ouyanghao')
assert src==pathlib.Path(s['checkpoint_dir'])/('vae_state_'+str(step)+'.pth') and src.resolve()==src and home in src.parents
for p in [src,*src.parents]:
 if p==home:assert p.stat().st_uid==1013;break
 assert p.lstat().st_uid==1013 and not p.is_symlink()
def fingerprint(st):return dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns,owner_uid=st.st_uid,nlink=st.st_nlink,mode=stat.S_IMODE(st.st_mode))
st=src.lstat();before=fingerprint(st)
assert stat.S_ISREG(st.st_mode) and st.st_uid==1013 and st.st_nlink==1 and st.st_size==expected['bytes']
assert time.time()-st.st_mtime>=30,'Wait for a real checkpoint stable for at least30s'
if prior is not None:assert before==prior['source_fingerprint']
h=hashlib.sha256();started=time.monotonic()
fd=os.open(str(src),os.O_RDONLY|os.O_NOFOLLOW)
with os.fdopen(fd,'rb') as f:
 assert fingerprint(os.fstat(f.fileno()))==before
 for block in iter(lambda:f.read(8*1024**2),b''):
  assert time.monotonic()-started<40,'Bounded source hashing timed out'
  h.update(block)
 assert fingerprint(os.fstat(f.fileno()))==before
assert fingerprint(src.lstat())==before and h.hexdigest()==expected['sha256']
print(json.dumps(dict(status='STABLE_FILTERED15000_CHECKPOINT_BYTES_VERIFIED',run='filtered15000_vae200k_train_v1',
 manager_step=step,source=expected,source_fingerprint=before,source_sha256=h.hexdigest(),spec_sha256=SPEC_LITERAL,
 runtime_binding_sha256=BINDING_LITERAL,producer_receipt_sha256=hashlib.sha256(recraw).hexdigest(),
 producer_observed_primitive_attempts=rec['completed_primitive_attempts'],producer_exact_reload_reported=proof,
 checkpoint_payload_deserialized=False,new_model_loads=0,new_gpu_contexts=0,new_sdk_tasks=0,signals_sent=False,
 source_modified=False,observed_unix=time.time(),source_hash_seconds=time.monotonic()-started)))
'''
 for k,v in [('ROOT_LITERAL',ROOT),('STEP_LITERAL',step),('EXPECTED_LITERAL',expected),('PRIOR_LITERAL',prior),('SPEC_LITERAL',SPEC_SHA),('BINDING_LITERAL',BINDING_SHA)]:code=code.replace(k,repr(v))
 ast.parse(code);return code

def main():
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True,choices=[50000,100000,150000,200000]);a=p.parse_args()
 assert os.name=='nt'
 sp=B/'configs/filtered15000-vae-spec.json';assert sha(sp)==SPEC_SHA
 spec=json.loads(sp.read_text(encoding='utf-8'))
 progress=B/'results/filtered15000_handoff_20261027'/RUN/'progress.json'
 ck=validate_collected(json.loads(progress.read_text(encoding='utf-8')),a.step,spec)
 guard_local(D,D,directory=True)
 lockpath=D/'filtered15000_vae_checkpoint_backup.lock';guard_local(D,lockpath)
 lock=lockpath.open('a+b')
 if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
 lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
 dest=D/'checkpoints/private/filtered15000_vae200k_20261028'/('step_'+str(a.step))
 guard_local(D,dest,directory=True,must_absent=True)
 assert shutil.disk_usage(D).free>=RESERVE+METADATA_ALLOWANCE+ck['bytes'],'Insufficient D backup space; preserve all earlier copies'
 dest.mkdir(parents=True,exist_ok=False);guard_local(D,dest,directory=True)
 E=B/'results/filtered15000_vae_checkpoint_backups_20261028'/('step_'+str(a.step))
 E.mkdir(parents=True,exist_ok=False)
 intent=dict(schema='textop.filtered15000_vae_private_byte_backup.v1',status='BACKUP_INTENT',run=RUN,checkpoint=ck,
  collected_progress_sha256=sha(progress),spec_sha256=SPEC_SHA,runtime_binding_sha256=BINDING_SHA,
  reserve_bytes=RESERVE,metadata_allowance_bytes=METADATA_ALLOWANCE,destination=str(dest),tool_sha256=sha(__file__))
 dump_new(E/'intent.json',intent);dump_new(dest/'intent.json',intent)
 try:
  proof=json.loads(recorded_remote(remote_script(a.step,ck),E/'source_before_client.json',timeout=75))
  dump_new(E/'source_before.json',proof);dump_new(dest/'source_before.json',proof)
  partial=dest/'vae_state.pth.partial';final=dest/('vae_state_'+str(a.step)+'.pth')
  guard_local(D,partial,must_absent=True);guard_local(D,final,must_absent=True)
  assert shutil.disk_usage(D).free>=RESERVE+METADATA_ALLOWANCE+ck['bytes']
  command=['scp','-B','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+ck['path'],str(partial)]
  v=recorded_client('',E/'scp_client.json',timeout=1200,command=command)
  assert v['actual_client_exit_code']==0 and v['client_reaped'] and not v['timed_out'] and v['streams_final'] and v['error'] is None,'Inspect the retained client and partial before any new attempt'
  guard_local(D,partial);assert partial.stat().st_size==ck['bytes'] and sha(partial)==ck['sha256']
  with partial.open('r+b') as f:f.flush();os.fsync(f.fileno())
  after=json.loads(recorded_remote(remote_script(a.step,ck,proof),E/'source_after_client.json',timeout=75))
  assert after['source_fingerprint']==proof['source_fingerprint'] and after['source_sha256']==ck['sha256']
  dump_new(E/'source_after.json',after);dump_new(dest/'source_after.json',after)
  assert shutil.disk_usage(D).free>=RESERVE+METADATA_ALLOWANCE
  guard_local(D,partial);guard_local(D,final,must_absent=True)
  partial.rename(final)
  guard_local(D,final);assert final.stat().st_size==ck['bytes'] and sha(final)==ck['sha256']
  result=dict(intent,status='PRIVATE_FILTERED15000_VAE_BYTE_BACKUP_VERIFIED',D_checkpoint=str(final),
   D_checkpoint_sha256=ck['sha256'],D_checkpoint_bytes=ck['bytes'],D_free_after_bytes=shutil.disk_usage(D).free,
   source_before_sha256=sha(E/'source_before.json'),source_after_sha256=sha(E/'source_after.json'),
   actual_scp_exit_code=0,client_streams_final=True,source_fingerprint_stable=True,source_server_copy_retained=True,
   producer_exact_reload_reported=proof['producer_exact_reload_reported'],independent_model_reload_performed=False,
   checkpoint_payload_deserialized=False,new_gpu_contexts=0,new_sdk_tasks=0,new_training_runs=0,signals_sent=False,
   files_deleted=0,published_full_state=False,scope='Private byte/provenance copy only, not a new gradient or CPU tensor audit',completed_unix=time.time())
  dump_new(dest/'backup.json',result);dump_new(E/'backup.json',result)
  assert shutil.disk_usage(D).free>=RESERVE
  print(json.dumps(dict(status=result['status'],step=a.step,bytes=ck['bytes'],sha256=ck['sha256'],D_free_bytes=result['D_free_after_bytes'])))
 except BaseException as e:
  failure=dict(status='BACKUP_INCOMPLETE_RETAIN_ALL_FILES_INSPECT_ORIGINAL_CLIENT',error=repr(e),step=a.step,observed_unix=time.time(),server_training_changed=False)
  dump_new(E/'failure.json',failure);dump_new(dest/'failure.json',failure);raise

if __name__=='__main__':main()
