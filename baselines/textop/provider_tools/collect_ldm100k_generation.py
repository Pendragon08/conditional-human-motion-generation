"""Live metadata/log tails, then terminal witness-array or diagnostic-byte audit."""
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
assert a.run in ['user_ldm100k_generation_author_v1', 'user_ldm100k_generation_document_v1', 'user_ldm100k_diagnostic32_v1']
assert a.evidence.isascii() and a.evidence.isidentifier()
code = r'''
import fcntl,hashlib,json,os,pathlib,time,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'));run=RUN;prefix='ldm100k_generation_'+run
from ldm100k_verified_input import read,sha,owned,closed_job
from ldm100k_generation_acceptance import validate_witness,runtime_signature
from ldm100k_generation_artifacts import audit_witness_artifacts,audit_diagnostic_journal
paths=dict(receipt=r/'ldm100k_user_handoff_20261028'/run/'receipt.json',**{k:r/'receipts'/(prefix+suffix) for k,suffix in [('exit','_exit.json'),('process','_process_exit.json'),('outer','_outer_exit.json'),('driver','_driver.json')]})
result=dict(kind='ldm100k_generation',run=run,observed_unix=time.time(),files={},sha256={},workers=[],closed_execution_verified=False,new_model_loads=0,new_gpu_contexts=0)
for key,p in paths.items():
 if p.is_file():result['files'][key]=read(p);result['sha256'][key]=sha(p)
names=[str(r/'scripts'/n) for n in ['run_ldm100k_heldout_generation.py','supervise_ldm100k_heldout_generation.py']]
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
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));result[suffix+'_tail']=f.read().decode(errors='replace')
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);result['common_lock_free']=True
except BlockingIOError:result['common_lock_free']=False
finally:lock.close()
files=result['files']
if all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0:
 sp=owned(r/'configs/ldm100k-generation-spec.json');s=read(sp);rec=files['receipt'];proof_spec=s
 if rec['mode']=='witness' and rec['spec_sha256']!=sha(sp):
  old=owned(r/'receipts/ldm100k_generation_spec_before_diagnostic.json');proof_spec=read(old)
  assert sha(old)==rec['spec_sha256'] and runtime_signature(proof_spec)==runtime_signature(s)
 else:assert rec['spec_sha256']==sha(sp)
 for n,h in proof_spec['bound_sources'].items():assert sha(owned(r/n))==h,n
 assert rec['source_sha256']==proof_spec['executor_sha256'] and rec['output_writers_closed']
 result['closure']=closed_job(prefix,paths['receipt'],result['sha256']['receipt'],proof_spec['supervisor_sha256'])
 assert not result['workers'] and result['common_lock_free']
 if rec['mode']=='witness':
  validate_witness(rec,run,proof_spec);result['artifact_audit']=audit_witness_artifacts(paths['receipt'].parent,rec)
 else:
  assert rec['status']=='USER_LDM100K_DIAGNOSTIC32_GENERATION_FINISHED' and rec['unattempted_pairs']==0
  result['artifact_audit']=audit_diagnostic_journal(paths['receipt'].parent,rec,proof_spec)
 result['runtime_binding_sha256']=runtime_signature(proof_spec);result['closed_execution_verified']=True
print(json.dumps(result))
'''.replace('RUN', repr(a.run))
ast.parse(code, feature_version=(3, 8))
out = B / 'results/ldm100k_user_handoff_20261028' / a.run
out.mkdir(exist_ok=True)
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
