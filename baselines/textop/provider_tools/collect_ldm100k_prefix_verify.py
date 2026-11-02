"""Read independent CPU verification receipts only; never deserialize arrays/checkpoints."""
import argparse
import ast
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--run', required=True)
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert all(x.isascii() and x.isidentifier() for x in [a.run, a.evidence])
code = r'''
import fcntl,hashlib,json,os,pathlib,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
run=RUN;prefix='ldm100k_prefix_verify_'+run
paths=dict(receipt=r/'ldm100k_user_handoff_20261028'/run/'receipt.json',**{k:r/'receipts'/(prefix+suffix) for k,suffix in [('exit','_exit.json'),('process','_process_exit.json'),('outer','_outer_exit.json'),('driver','_driver.json')]})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(kind='ldm100k_prefix_verify',run=run,observed_unix=time.time(),files={},sha256={},workers=[],closed_execution_verified=False,arrays_read=0,model_loads=0)
for key,p in paths.items():
 if p.is_file():result['files'][key]=json.loads(p.read_bytes());result['sha256'][key]=sha(p)
names=[str(r/'scripts'/n) for n in ['verify_ldm100k_prefix_export.py','supervise_ldm100k_prefix_verify.py']]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0');st=(proc/'stat').read_text().rsplit(')',1)[1].split()
  if any(n in argv for n in names):result['workers'].append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
for suffix in ['.log','_outer.log']:
 p=r/'logs'/(prefix+suffix)
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));result[suffix+'_tail']=f.read().decode(errors='replace')
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result['common_lock_free']=True
except BlockingIOError:result['common_lock_free']=False
finally:lock.close()
files=result['files']
if all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0:
 from sys import path;path.insert(0,str(r/'scripts'))
 from filtered15000_vae_completion import closed_job
 s=json.loads((r/'configs/ldm100k-prefix-verify-spec.json').read_bytes());rec=files['receipt']
 assert rec['status']=='USER_LDM100K_PREFIX_FRESH_CPU_VERIFIED' and rec['output_writers_closed']
 assert rec['spec_sha256']==sha(r/'configs/ldm100k-prefix-verify-spec.json')
 assert rec['source_sha256']==s['executor_sha256'] and rec['fullstate_and_exported_denoiser_exact'] and rec['cpu_adam_restore_exact'] and rec['synthetic_cpu_forward_exact']
 assert rec['training_optimizer_updates_performed']==0 and rec['new_gpu_contexts']==0
 assert rec['original300k_completion_claimed'] is False and rec['semantic_quality_accepted'] is False
 result['closure']=closed_job(prefix,paths['receipt'],result['sha256']['receipt'],s['supervisor_sha256'])
 assert not result['workers'] and result['common_lock_free'];result['closed_execution_verified']=True
print(json.dumps(result))
'''.replace('RUN', repr(a.run))
ast.parse(code, feature_version=(3, 8))
out = B / 'results/ldm100k_user_handoff_20261028' / a.run
out.mkdir(parents=True, exist_ok=True)
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=90))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
