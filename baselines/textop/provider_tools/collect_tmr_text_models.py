"""Read-only public text-model preparation closure, signatures and frozen CPU probe."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
out = B / 'results/semantic_evaluator_20261029/tmr_text_models_v1'
code = r'''
import json,pathlib,os,sys,fcntl,time,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import read,sha,owned
run='tmr_text_models_v1';prefix='tmr_text_models_'+run
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
  if any(str(r/'scripts'/n) in argv for n in ['prepare_tmr_text_models.py','supervise_tmr_text_models.py']):
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
 rec=files['receipt'];assert rec['status']=='PINNED_TMR_TEXT_MODELS_CPU_VERIFIED' and rec['output_writers_closed']
 assert rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(sp)==files['exit']['request_sha256']
 assert hashes['process']==files['exit']['process_exit_sha256'] and sha(log/'resources.jsonl')==files['exit']['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and not samples[-1]['owned_group_processes']
 for item in rec['completed_files']:
  p=owned(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 assert sum(x['bytes'] for x in rec['completed_files'])==rec['total_verified_public_file_bytes']==707323471
 assert read(parent/run/'probe_exit.json')['actual_returncode']==0
 probe=rec['capability_probe'];assert len(probe['text_models'])==2 and not probe['cuda_initialized']
 assert all(x['finite'] and x['frozen'] and x['tokens'][0]==2 and x['tokens'][2]==768 for x in probe['text_models'])
 assert probe['actual_training_updates']==probe['private_motion_or_babel_text_reads']==0
 assert not probe['robot_semantic_encoder_trained'] and not probe['formal_tmr_metrics_available']
 assert rec['robot_motion_or_babel_text_reads']==rec['learned_tmr_weight_reads']==rec['training_optimizer_updates']==0
 audit=dict(all_pinned_public_text_model_bytes_and_hashes_verified=True,frozen_offline_variable_length_cpu_forward_verified=True,
  actual_evaluator_training_updates=0,trained_semantic_metrics_available=False)

print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,
 closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,data_free_bytes=shutil.disk_usage(parent).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)))
'''
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
