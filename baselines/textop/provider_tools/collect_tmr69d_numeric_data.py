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
out = B / 'results/semantic_evaluator_20261029/tmr69d_numeric_data_v1'
code = r'''
import json,pathlib,os,sys,fcntl,time,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import read,sha,owned
run='tmr69d_numeric_data_v1';prefix='tmr_numeric_data_'+run
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
  if any(str(r/'scripts'/n) in argv for n in ['build_tmr69d_numeric_data_v1.py','supervise_tmr69d_numeric_data_v1.py']):
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
 rec=files['receipt'];assert rec['status']=='TMR69D_NUMERIC_TRAIN_INPUTS_ACCEPTED' and rec['output_writers_closed']
 assert rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(sp)==files['exit']['request_sha256']
 assert hashes['process']==files['exit']['process_exit_sha256'] and sha(log/'resources.jsonl')==files['exit']['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and not samples[-1]['owned_group_processes']
 item=rec['private_artifact'];p=owned(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 rows=read(p);assert len(rows)==rec['original_train_sources']==5361
 scope=read(s['scope_receipt']['path']);assert sha(s['scope_receipt']['path'])==s['scope_receipt']['sha256']
 data=read(scope['private_artifact']['path']);sources={x['canonical_source']:x for x in data['sources']}
 assert len(rows)==len({x['canonical_source'] for x in rows})==len(sources)
 for x in rows:
  prior=sources[x['canonical_source']];assert prior['original_split']=='train' and prior['mirrored'] is False
  assert x['source_sha256']==prior['output_sha256'] and x['path']==prior['output_file'] and x['split']==prior['split']
  assert x['feature_frames']==x['pose_frames']-1==prior['frames50']-1 and x['finite'] and x['unit_quaternion']
  assert x['normalized_feature_bytes']==x['feature_frames']*69*4
 assert sum(x['feature_frames'] for x in rows)==rec['actual_unique_source_feature_frames']
 assert rec['full_normalized_feature_RAM_bytes']==sum(x['normalized_feature_bytes'] for x in rows)
 assert rec['original_validation_sources_read']==rec['encoder_weight_loads']==rec['sdk_simulation_model_constructions']==rec['physics_steps']==rec['optimizer_updates']==0
 assert rec['actual_sample_windows_verified']==len(data['samples']) and not rec['cuda_initialized']
 assert rec['variable_length_probe']['finite'] and rec['variable_length_probe']['padded_exact_zero']
 assert not rec['normalized_features_persisted'] and not rec['complete_evaluator_training_ready']
 audit=dict(all5361_private_numeric_source_rows_and_exact_scope_join_verified=True,all_segment_and_feature_frame_counts_verified=True,
  actual_original_validation_sources_read=0,actual_learned_encoder_loads=0,actual_training_updates=0,complete_evaluator_training_ready=False)

print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,
 closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,data_free_bytes=shutil.disk_usage(parent).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)))
'''
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
