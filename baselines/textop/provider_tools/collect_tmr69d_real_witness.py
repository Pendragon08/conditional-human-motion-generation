"""Read-only actual real-data TMR witness closure and cross-process comparison."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--role',choices=['author','document'],required=True)
p.add_argument('--evidence',required=True)
a=p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
run='tmr69d_real_witness_'+a.role+'_v1'
out=B/'results/semantic_evaluator_20261029'/run
code=r'''
import os,sys,pathlib,json,fcntl,time,shutil,math
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import owned,read,sha
run=RUN_LITERAL;prefix='tmr_real_witness_'+run;parent=owned(r/'evaluator_tmr_20261029');log=parent/(run+'_supervision');sp=owned(r/'receipts'/(prefix+'_request.json'));s=read(sp)
files={};hashes={}
for key,path in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if path.exists():files[key]=read(path);hashes[key]=sha(path)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['run_tmr69d_real_witness.py','supervise_tmr69d_real_witness.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (ProcessLookupError,FileNotFoundError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_reaped'] and files['process']['child_exit_code']==0 and not workers and free
audit=None
if closed:
 rec=files['receipt'];assert rec['status']=='TMR69D_REAL_DATA_GPU_WITNESS_VERIFIED' and rec['output_writers_closed']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(sp)==files['exit']['request_sha256'] and hashes['process']==files['exit']['process_exit_sha256']
 assert sha(log/'resources.jsonl')==files['exit']['resource_sha256'] and rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 resources=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()];assert resources[-1]['phase']=='after_exit' and not resources[-1]['owned_group_processes']
 assert rec['actual_diagnostic_optimizer_calls']==4 and rec['selected_logical_optimizer_steps']==3
 assert rec['actual_formal_evaluator_updates']==rec['generator_or_teacher_updates']==rec['persistent_fullstate_writes']==rec['actual_validation_optimizer_updates']==0
 assert rec['actual_real_train_source_families']==5361 and rec['actual_original_validation_source_reads']==0
 assert rec['exact_model_adamw_all_rng_sampler_nextbatch_loss_replay'] and rec['validation_restored_training_rng'] and rec['source_arrays_norm_and_public_model_fingerprints_unchanged']
 assert rec['original_class_method_AST_unchanged'] and all(x['original_class_AST_unchanged'] for x in rec['original_class_AST_audit'].values())
 assert rec['actual_losses'][2]==rec['actual_losses'][3] and rec['actual_batch_sha256'][2]==rec['actual_batch_sha256'][3]
 assert all(math.isfinite(v) for row in rec['actual_losses'] for v in row.values())
 compared=False
 if s['role']=='document':
  item=s['author_receipt'];assert sha(owned(item['path']))==item['sha256'];author=read(item['path'])
  for key in ['runtime_binding_sha256','actual_input_signature','actual_input_signature_sha256','actual_initial_model_sha256','actual_final_model_sha256','final_model_adamw_rng_sampler_content_sha256','actual_losses','actual_batch_sha256','internal_validation_loss','internal_validation_batch_sha256']:
   assert rec[key]==author[key],key
  compared=True
 audit=dict(real_data_fullstate_replay_and_zero_validation_updates_verified=True,original_TMR_method_AST_unchanged=True,
  cross_process_real_input_model_adamw_rng_loss_exact=compared,actual_diagnostic_optimizer_calls=4,actual_formal_evaluator_updates=0,
  diagnostic_weights_not_formal_initialization=True,formal_semantic_metrics_available=False)
logs={}
for path in [log/'child.log',parent/(run+'_outer.log')]:
 if path.exists():
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-2500));logs[path.name]=f.read().decode(errors='replace')
print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,data_free_bytes=shutil.disk_usage(parent).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)))
'''.replace('RUN_LITERAL',repr(run))
j=json.loads(recorded_remote(code,out/(a.evidence+'_client.json'),timeout=180))
(out/(a.evidence+'.json')).write_text(json.dumps(j,indent=2),encoding='utf-8')
(out/'progress.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))
