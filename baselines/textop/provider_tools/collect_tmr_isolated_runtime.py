"""Read-only private runtime bootstrap closure, hashes and actual CPU probe."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
out = B / 'results/semantic_evaluator_20261029/tmr_isolated_runtime_v1'
code = r'''
import json,pathlib,os,sys,fcntl,time,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import read,sha,owned
run='tmr_isolated_runtime_v1';prefix='tmr_isolated_runtime_'+run
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
  if any(str(r/'scripts'/n) in argv for n in ['bootstrap_tmr_isolated_runtime.py','supervise_tmr_isolated_runtime.py']):
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
 rec=files['receipt'];assert rec['status']=='TMR_ISOLATED_RUNTIME_BOOTSTRAP_FINISHED' and rec['output_writers_closed']
 assert rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(sp)==files['exit']['request_sha256']
 assert hashes['process']==files['exit']['process_exit_sha256'] and sha(log/'resources.jsonl')==files['exit']['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and not samples[-1]['owned_group_processes']
 for item in rec['completed_wheels']:
  p=owned(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 assert len(rec['completed_wheels'])==6 and read(parent/run/'pip_exit.json')['actual_returncode']==0
 assert read(parent/run/'probe_exit.json')['actual_returncode']==0
 probe=rec['capability_probe'];assert probe['actual_cpu_random_architecture_constructions']==3
 assert probe['variable_length_forward_finite'] and probe['padded_decoder_output_exact_zero'] and not probe['cuda_initialized']
 assert probe['pretrained_weight_reads']==probe['actual_motion_or_text_data_reads']==probe['optimizer_updates']==0
 assert not probe['formal_training_ready'] and rec['original_module_snapshot_before']==rec['original_module_snapshot_after']
 assert sha(owned(s['actor_path']))==s['actor_sha256']
 assert rec['model_weight_payload_reads']==rec['motion_array_reads']==rec['raw_text_reads']==rec['formal_training_updates']==0
 audit=dict(six_pinned_public_wheels_sha256_verified=True,private_target_imports_and_finite_variable_length_cpu_forward_verified=True,
  original_module_source_snapshot_unchanged=True,actual_evaluator_training_updates=0,trained_semantic_metrics_available=False)
print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,
 closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,data_free_bytes=shutil.disk_usage(parent).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)))
'''
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
