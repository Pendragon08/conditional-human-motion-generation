"""Read-only actual finite69D numeric inputs with complete TRAIN source/hash coverage."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
out = B / 'results/semantic_evaluator_20261029/tmr69d_core_contract_v1'
code = r'''
import json,pathlib,os,sys,fcntl,time,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import read,sha,owned
run='tmr69d_core_contract_v1';prefix='tmr_core_contract_'+run
sp=owned(r/'receipts'/(prefix+'_request.json'));s=read(sp);parent=owned(r/'evaluator_tmr_20261029');log=parent/(run+'_supervision')
files={};hashes={}
for k,p in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if p.exists():files[k]=read(p);hashes[k]=sha(p)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['build_tmr69d_core_contract_v1.py','supervise_tmr69d_core_contract_v1.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
logs={}
for p in [log/'child.log',parent/(run+'_outer.log')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3000));logs[p.name]=f.read().decode(errors='replace')
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_exit_code']==0 and not workers and free
audit=None
if closed:
 rec=files['receipt'];assert rec['status']=='TMR69D_SYNTHETIC_CPU_CORE_CONTRACT_VERIFIED' and rec['output_writers_closed']
 assert rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(sp)==files['exit']['request_sha256']
 assert hashes['process']==files['exit']['process_exit_sha256'] and sha(log/'resources.jsonl')==files['exit']['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and not samples[-1]['owned_group_processes']
 assert rec['actual_diagnostic_optimizer_calls']==4 and rec['selected_logical_optimizer_steps']==3
 assert rec['actual_formal_evaluator_updates']==rec['real_robot_or_babel_text_reads']==rec['pretrained_weight_payload_reads']==rec['persistent_checkpoint_writes']==0
 assert rec['exact_model_adamw_all_rng_sampler_nextbatch_loss_replay'] and rec['original_method_bodies_unchanged']
 assert all(x['original_class_AST_unchanged'] for x in rec['original_class_AST_audit'].values())
 assert rec['optimizer']==dict(type='torch.optim.AdamW',lr=.0001,weight_decay=.01)
 assert rec['actual_losses'][2]==rec['actual_losses'][3]
 assert not rec['cuda_initialized'] and rec['diagnostic_weights_discarded_not_formal_initialization']
 assert not rec['real_data_or_semantic_acceptance_claimed']
 audit=dict(actual_original_method_AST_and_cpu_full_recovery_contract_verified=True,
  actual_cpu_diagnostic_optimizer_calls=4,selected_logical_optimizer_steps=3,actual_formal_evaluator_updates=0,
  actual_real_robot_or_text_reads=0,actual_pretrained_model_reads=0,real_data_training_or_semantic_ready=False)

print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,
 closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,data_free_bytes=shutil.disk_usage(parent).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)))
'''
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
