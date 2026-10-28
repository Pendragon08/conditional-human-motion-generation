"""Light live evaluator receipts; terminal immutable fullstate-byte closure."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--evidence',required=True)
a=p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
out=B/'results/semantic_evaluator_20261029/tmr69d_evaluator500_train_v1'
code=r'''
import os,sys,pathlib,json,fcntl,time,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import owned,read,sha
run='tmr69d_evaluator500_train_v1';prefix='tmr_evaluator_train_'+run;parent=owned(r/'evaluator_tmr_20261029');log=parent/(run+'_supervision');sp=owned(r/'configs/tmr69d-evaluator-train-spec.json');s=read(sp)
files={};hashes={}
for key,path in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if path.exists():files[key]=read(path);hashes[key]=sha(path)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['train_tmr69d_evaluator.py','supervise_tmr69d_evaluator_training.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (ProcessLookupError,FileNotFoundError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_reaped'] and files['process']['child_exit_code']==0 and not workers and free
audit=None
if closed:
 rec=files['receipt'];assert rec['status']=='TMR69D_FORMAL_EVALUATOR_500_EPOCHS_FINISHED' and rec['output_writers_closed']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(sp)==files['exit']['request_sha256'] and hashes['process']==files['exit']['process_exit_sha256']
 assert sha(log/'resources.jsonl')==files['exit']['resource_sha256'] and rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 resources=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()];assert resources[-1]['phase']=='after_exit' and not resources[-1]['owned_group_processes']
 assert rec['formal_optimizer_updates']==rec['attempted_optimizer_updates']==507500 and rec['completed_epochs']==500 and rec['skipped_updates']==0
 assert rec['generator_or_teacher_training_updates']==rec['original_validation_sources_used_for_gradient']==rec['internal_validation_optimizer_updates']==0
 assert rec['exact_final_model_adamw_all_rng_sampler_nextbatch_disk_reload'] and not rec['inference_export_and_heldout_semantic_metrics_accepted']
 assert [x['epoch'] for x in rec['checkpoints']]==[100,500] and all(not x['emergency'] for x in rec['checkpoints'])
 for item in rec['checkpoints']:
  p=owned(item['path'],pathlib.Path('/home/ouyanghao'));assert p.parent==pathlib.Path(s['checkpoint_dir'])
  assert p.stat().st_size==item['bytes']<=201326592 and sha(p)==item['sha256']
  assert item['exact_fullstate_deserialization'] and item['production_model_adamw_rng_sampler_nextbatch_reloaded'] and not item['pending_validation']
 assert rec['selected_checkpoint']==rec['checkpoints'][-1]
 audit=dict(two_fullstate_bytes_sha256_and_closed_500_epoch_counters_verified=True,actual_formal_evaluator_updates=507500,
  actual_generator_or_teacher_new_updates=0,actual_original_validation_gradient_reads=0,model_payloads_deserialized_by_collector=0,
  inference_and_semantic_metrics_still_pending=True)
logs={}
for path in [log/'child.log',parent/(run+'_outer.log')]:
 if path.exists():
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-2000));logs[path.name]=f.read().decode(errors='replace')
last_resource=None
if (log/'resources.jsonl').exists():
 with (log/'resources.jsonl').open('rb') as f:f.seek(max(0,(log/'resources.jsonl').stat().st_size-8000));last_resource=json.loads(f.read().decode().splitlines()[-1])
print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,last_resource=last_resource,data_free_bytes=shutil.disk_usage(parent).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)))
'''
j=json.loads(recorded_remote(code,out/(a.evidence+'_client.json'),timeout=180))
(out/(a.evidence+'.json')).write_text(json.dumps(j,indent=2),encoding='utf-8')
(out/'progress.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print(json.dumps(j))
