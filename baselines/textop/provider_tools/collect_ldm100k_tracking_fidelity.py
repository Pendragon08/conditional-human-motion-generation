"""Light live FK metrics; independent terminal scalar/coverage aggregation."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
run = 'user_ldm100k_tracking_fidelity_v1'
out = B / 'results/ldm100k_user_handoff_20261028' / run
code = r'''
import json,pathlib,os,sys,fcntl,time,shutil,math
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'));from ldm100k_verified_input import read,sha,owned
run='user_ldm100k_tracking_fidelity_v1';prefix='ldm100k_tracking_fidelity_'+run
sp=owned(r/'receipts'/(prefix+'_request.json'));s=read(sp);parent=owned(s['output_root'],pathlib.Path('/home/ouyanghao'));log=parent/(run+'_supervision')
files={};hashes={}
for k,p in dict(receipt=parent/run/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if p.exists():files[k]=read(p);hashes[k]=sha(p)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['measure_ldm100k_tracking_fidelity.py','supervise_ldm100k_tracking_fidelity.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
logs={}
for p in [log/'child.log',parent/(run+'_outer.log')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2000));logs[p.name]=f.read().decode(errors='replace')
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_exit_code']==0 and not workers and free
audit=None
if closed:
 rec=files['receipt'];assert rec['status']=='USER_LDM100K_RECORDED_TRACKING_FIDELITY_FINISHED' and rec['output_writers_closed']
 assert rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(sp)==files['exit']['request_sha256']
 assert hashes['process']==files['exit']['process_exit_sha256'] and sha(log/'resources.jsonl')==files['exit']['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and not samples[-1]['owned_group_processes']
 artifact=rec['private_artifact'];p=owned(artifact['path'],pathlib.Path('/home/ouyanghao'));assert sha(p)==artifact['sha256'] and p.stat().st_size==artifact['bytes']
 rows=read(p);assert len(rows)==rec['processed_trials']==1880
 pr=read(s['physics_binding']['receipt']['path']);index=pathlib.Path(s['physics_binding']['receipt']['path']).parent/'executed_traces.private.json'
 assert sha(index)==pr['private_executed_trace_index_sha256'];prior={(x['segment_id'],x['seed'],x['mode']):x for x in read(index)}
 seen=set()
 for x in rows:
  key=x['segment_id'],x['seed'],x['mode'];assert key not in seen and key in prior;seen.add(key)
  assert x['physical_failure']==prior[key]['physical_failure'] and x['full_T_minus_1_coverage']==prior[key]['final_reference_endpoint_position']['full_T_minus_1_coverage']
  assert x['executed_control_steps']==prior[key]['samples']-1
  for v in x['groups'].values():
   assert v['full_coverage_and_within_threshold']==bool(x['full_T_minus_1_coverage'] and v['max_frame_root_relative_mean_m']<=.3)
   assert all(math.isfinite(v[k]) and v[k]>=0 for k in ['observed_global_mpjpe_mm','observed_root_relative_mpjpe_mm','max_frame_root_relative_mean_m'])
 assert seen==set(prior)
 for mode in ['conditional_cfg5','unconditional']:
  rr=[x for x in rows if x['mode']==mode];assert len(rr)==940
  for group in ['tracked14','all30']:
   values=[x['groups'][group] for x in rr];got=rec['aggregate'][mode][group]
   assert got['full_coverage_successes']==sum(v['full_coverage_and_within_threshold'] for v in values)
   assert math.isclose(got['trial_equal_global_mpjpe_mm'],sum(v['observed_global_mpjpe_mm'] for v in values)/940,rel_tol=1e-12,abs_tol=1e-10)
   assert math.isclose(got['trial_equal_root_relative_mpjpe_mm'],sum(v['observed_root_relative_mpjpe_mm'] for v in values)/940,rel_tol=1e-12,abs_tol=1e-10)
 assert rec['new_physics_steps']==rec['new_policy_rollouts']==rec['neural_model_loads']==rec['optimizer_updates']==0
 audit=dict(all1880_private_scalar_rows_and_original_coverage_join_verified=True,threshold_and_aggregate_recomputed=True,source_physics_decisions_unchanged=True,new_model_loads=0,new_gpu_contexts=0)
print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,home_free_bytes=shutil.disk_usage(parent).free)))
'''
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=180))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
