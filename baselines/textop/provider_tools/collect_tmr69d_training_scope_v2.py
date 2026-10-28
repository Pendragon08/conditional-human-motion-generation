"""Read-only TRAIN-only metadata scope closure and independent family/segment coverage."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
out = B / 'results/semantic_evaluator_20261029/tmr69d_training_scope_v2'
code = r'''
import json,pathlib,os,sys,fcntl,time,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import read,sha,owned
run='tmr69d_training_scope_v2';prefix='tmr_training_scope_'+run
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
  if any(str(r/'scripts'/n) in argv for n in ['build_tmr69d_training_scope_v2.py','supervise_tmr69d_training_scope_v2.py']):
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
 rec=files['receipt'];assert rec['status']=='TMR69D_TRAINING_METADATA_SCOPE_BUILT' and rec['output_writers_closed']
 assert rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(sp)==files['exit']['request_sha256']
 assert hashes['process']==files['exit']['process_exit_sha256'] and sha(log/'resources.jsonl')==files['exit']['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and not samples[-1]['owned_group_processes']
 item=rec['private_artifact'];p=owned(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 data=read(p);assert data['summary']==rec['summary']
 sources={x['canonical_source']:x for x in data['sources']};assert len(sources)==5361
 assert all(x['original_split']=='train' and x['mirrored'] is False for x in sources.values())
 assert sum(x['split']=='gradient_train' for x in sources.values())==5093
 assert sum(x['split']=='internal_validation' for x in sources.values())==268
 ids=set()
 for x in data['samples']:
  assert x['sample_id'] not in ids;ids.add(x['sample_id'])
  assert x['split']==sources[x['canonical_source']]['split'] and x['original_split']=='train'
  assert x['text_id'] in data['texts'] and x['mirrored'] is False
  assert 0<=x['start_feature']<x['end_feature_exclusive']<=sources[x['canonical_source']]['frames50']-1
  assert x['feature_frames']==x['end_feature_exclusive']-x['start_feature']<=200
 assert len(ids)==rec['summary']['samples']
 assert sha(owned(data['normalization']['path'],__import__('pathlib').Path('/dev/shm/ouyanghao_textop_20261025')))==data['normalization']['sha256'] if data['normalization']['path'].startswith('/dev/shm/') else sha(owned(data['normalization']['path']))==data['normalization']['sha256']
 assert rec['actual_model_loads']==rec['actual_array_deserializations']==rec['optimizer_updates']==0
 assert rec['training_authorized'] is False and not rec['original_validation_families_used_for_training']
 audit=dict(actual_private_scope_family_partition_and_all_segment_boundaries_verified=True,source_and_text_identifiers_unique=True,
  actual_array_deserializations=0,actual_model_loads=0,actual_training_updates=0,numeric_data_or_training_ready=False)

print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,
 closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,data_free_bytes=shutil.disk_usage(parent).free,home_free_bytes=shutil.disk_usage('/home/ouyanghao').free)))
'''
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
