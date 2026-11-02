"""One source-bound witness retry after the real closed constructor failure."""
import ast
import base64
import hashlib
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
O = B / 'results/ldm100k_user_handoff_20261028'
archive = json.loads((B / 'sources/ldm100k_generation_failed_author_v1/source_bytes.json').read_bytes())
sp = B / 'configs/ldm100k-generation-spec.json'
old = sp.read_bytes()
assert hashlib.sha256(old).hexdigest() == archive['configs/ldm100k-generation-spec.json']['sha256']
assert hashlib.sha256(old).hexdigest() == 'fd23a3332a783d66105e1a8d6346a42055090b1858e4c406a491a9d780a93f19'
failure = json.loads((O / 'user_ldm100k_generation_author_v1/progress.json').read_bytes())
assert failure['files']['receipt']['error'] == "KeyError('vae_checkpoint_sha256')"
assert not failure['workers'] and failure['common_lock_free']
names = ['scripts/ldm100k_diagnostic_runtime.py', 'scripts/ldm100k_generation_acceptance.py',
         'scripts/run_ldm100k_heldout_generation.py']
sources = {}
for name in names:
    raw = (B / name).read_bytes()
    ast.parse(raw.decode('utf-8'), feature_version=(3, 8))
    sources[name] = dict(old_sha256=archive[name]['sha256'],
                         new_sha256=hashlib.sha256(raw).hexdigest(),
                         new_base64=base64.b64encode(raw).decode())
previous = base64.b64decode(archive[names[0]]['base64']).decode('utf-8')
expected = previous.replace("self.source['vae_checkpoint_sha256']",
                            "accepted['export']['stopped']['policy']['vae_checkpoint_sha256']")
assert expected == (B / names[0]).read_text(encoding='utf-8')

code = r'''
import ast,base64,fcntl,hashlib,json,os,pathlib,subprocess,sys,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
os.sched_setaffinity(0,{0,1,2,3});sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import read,sha,owned
from ldm100k_scope_acceptance import accepted_scope
from contact_filter_resources import sample_resources
sp=r/'configs/ldm100k-generation-spec.json';assert sha(sp)==OLD_SHA;s=read(sp);payload=SOURCES
lock=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert s['witnesses_authorized'] and not s['generation_authorized'] and s['witness_bindings'] is None
run='user_ldm100k_generation_author_v1';prefix='ldm100k_generation_'+run
out=r/'ldm100k_user_handoff_20261028'/run;rp=owned(out/'receipt.json')
rec=read(rp);ex=read(r/'receipts'/(prefix+'_exit.json'));pr=read(r/'receipts'/(prefix+'_process_exit.json'));outer=read(r/'receipts'/(prefix+'_outer_exit.json'))
assert rec['error']=="KeyError('vae_checkpoint_sha256')" and rec['status']=='FAILED_LDM100K_HELDOUT_GENERATION'
assert rec['spec_sha256']==sha(sp) and rec['completed_pairs']==rec['completed_trials']==rec['failed_pairs']==0 and rec['unattempted_pairs']==1
assert rec['training_optimizer_updates']==0 and outer['actual_returncode']==1
for item in [ex,pr]:
 assert item['child_exit_code']==1 and item['child_reaped'] and not item['timed_out'] and item['supervisor_error'] is None and item['sample_errors']==[]
assert ex['receipt_sha256']==sha(rp) and ex['process_exit_sha256']==sha(r/'receipts'/(prefix+'_process_exit.json'))
resources=r/'logs'/(prefix+'_resources.jsonl');rows=[json.loads(x) for x in resources.read_text().splitlines()]
assert ex['resource_sha256']==sha(resources) and len(rows)==ex['samples_written']==pr['samples_written'] and rows[-1]['phase']=='after_exit'
assert not rows[-1]['owned_group_processes']
assert sorted(p.name for p in out.iterdir())==['receipt.json']
workers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if p.stat().st_uid!=1013:continue
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['run_ldm100k_heldout_generation.py','supervise_ldm100k_heldout_generation.py']):workers.append(int(p.name))
 except (OSError,ProcessLookupError):pass
assert not workers
for name,h in s['bound_sources'].items():assert sha(owned(r/name))==h,name
assert sha(r/'scripts/run_ldm100k_heldout_generation.py')==s['executor_sha256']
scope=accepted_scope(s['scope_binding'])
assert scope['verified']['export']['stopped']['policy']['vae_checkpoint_sha256']=='99017f1cbd13cf2904bc5f87904e5a488a5b3ee1f920da1882f90fdad04029ed'
res=sample_resources(r,None,s['minimum_ram_free_bytes'],s['minimum_disk_free_bytes'])
assert res['data_free_bytes']>=s['reserve_bytes']+s['log_allowance_bytes']+s['accepted_output_upper_bytes']['witness']
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
assert any(int(x.split(',')[0])==1 and int(x.split(',')[1])>=20*1024 for x in gpus.splitlines())
saved=r/'receipts/ldm100k_generation_failed_author_v1';saved.mkdir(mode=0o700,exist_ok=False)
for name in ['configs/ldm100k-generation-spec.json']+list(payload):
 src=owned(r/name);dest=saved/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if name in payload:assert sha(src)==payload[name]['old_sha256']
 with dest.open('xb') as f:f.write(src.read_bytes());f.flush();os.fsync(f.fileno())
failure_proof=dict(receipt_sha256=sha(rp),exit_sha256=sha(r/'receipts'/(prefix+'_exit.json')),process_sha256=sha(r/'receipts'/(prefix+'_process_exit.json')),outer_sha256=sha(r/'receipts'/(prefix+'_outer_exit.json')),resource_sha256=sha(resources),actual_child_exit=1,workers=workers,last_resource=rows[-1],actual_generated_pairs=0,constructor_reached_cpu_payload_loads_and_cuda_context_before_error=True)
for name,item in payload.items():
 raw=base64.b64decode(item['new_base64']);assert hashlib.sha256(raw).hexdigest()==item['new_sha256'];ast.parse(raw.decode())
 p=r/name;temp=p.with_name(p.name+'.binding-repair.partial')
 with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 temp.replace(p)
 if name in s['bound_sources']:s['bound_sources'][name]=item['new_sha256']
s['executor_sha256']=payload['scripts/run_ldm100k_heldout_generation.py']['new_sha256']
s['inference_contract_revision']=2
assert s['witnesses_authorized'] and not s['generation_authorized'] and s['witness_bindings'] is None
raw=(json.dumps(s,indent=2)+'\n').encode();temp=sp.with_name(sp.name+'.binding-repair.partial')
with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
temp.replace(sp)
result=dict(status='FAILED_GENERATION_CONSTRUCTOR_BINDING_REPAIRED_V2_WITNESSES_ENABLED_NOT_DISPATCHED',observed_unix=time.time(),new_spec_sha256=sha(sp),new_spec_base64=base64.b64encode(raw).decode(),failure_proof=failure_proof,changed_sources={n:x['new_sha256'] for n,x in payload.items()},preflight_resources=res,vae_binding_source='actual stopped policy, already independently verified',actual_generated_pairs=0,new_model_loads=0,new_gpu_contexts=0,original_training_or_model_modified=False)
with (saved/'repair.json').open('x') as f:json.dump(result,f,indent=2)
lock.close();print(json.dumps(result))
'''.replace('OLD_SHA', repr(hashlib.sha256(old).hexdigest())).replace('SOURCES', repr(sources))
ast.parse(code, feature_version=(3, 8))
result = json.loads(recorded_remote(code, O / 'generation_binding_repair_v2_client.json', timeout=180))
raw = base64.b64decode(result['new_spec_base64'])
assert hashlib.sha256(raw).hexdigest() == result['new_spec_sha256']
sp.write_bytes(raw)
(O / 'generation_binding_repair_v2.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'new_spec_base64'}))
