"""Light live strict replay; independent terminal trial/pose/journal audit."""
import argparse
import json
import pathlib
from backup_filter_recovery_checkpoint import recorded_remote

B = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--evidence', required=True)
a = p.parse_args()
assert a.evidence.isascii() and a.evidence.isidentifier()
run = 'user_ldm100k_generated_physics_v1'
out = B / 'results/ldm100k_user_handoff_20261028' / run
code = r'''
import json,pathlib,os,fcntl,sys,time,shutil,struct,math
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013
sys.path.insert(0,str(r/'scripts'))
from ldm100k_verified_input import owned,read,sha
from strict_tracking_coverage import verify_trial
run='user_ldm100k_generated_physics_v1';prefix='ldm100k_generated_physics_'+run
request=owned(r/'receipts'/(prefix+'_request.json'));s=read(request);parent=owned(s['output_root'],pathlib.Path('/home/ouyanghao'))
log=parent/(run+'_supervision');folder=parent/run;files={};hashes={}
for key,p in dict(receipt=folder/'receipt.json',exit=log/'exit.json',process=log/'process_exit.json',outer=r/'receipts'/(prefix+'_outer_exit.json')).items():
 if p.exists():files[key]=read(p);hashes[key]=sha(p)
workers=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:
  if proc.stat().st_uid!=1013:continue
  argv=(proc/'cmdline').read_bytes().decode().split('\0')
  if any(str(r/'scripts'/n) in argv for n in ['evaluate_ldm100k_generated_physics.py','supervise_ldm100k_generated_physics.py']):
   st=(proc/'stat').read_text().rsplit(')',1)[1].split();workers.append(dict(pid=int(proc.name),start_ticks=st[19],state=st[0],argv=argv))
 except (FileNotFoundError,ProcessLookupError):pass
lock=(r/'receipts/full_library_capacity.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
except BlockingIOError:free=False
finally:lock.close()
logs={}
for p in [log/'child.log',parent/(run+'_outer.log')]:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));logs[p.name]=f.read().decode(errors='replace')
closed=all(k in files for k in ['receipt','exit','process','outer']) and files['outer']['actual_returncode']==0 and files['exit']['supervision_passed'] and files['process']['child_exit_code']==0 and not workers and free
audit=None
if closed:
 rec=files['receipt'];assert rec['status']=='USER_LDM100K_GENERATED_STRICT_PHYSICS_FINISHED' and rec['output_writers_closed']
 assert rec['source_sha256']==s['executor_sha256'] and files['exit']['source_sha256']==s['supervisor_sha256']
 assert hashes['receipt']==files['exit']['receipt_sha256'] and sha(request)==files['exit']['request_sha256']
 assert hashes['process']==files['exit']['process_exit_sha256'] and sha(log/'resources.jsonl')==files['exit']['resource_sha256']
 samples=[json.loads(x) for x in (log/'resources.jsonl').read_text().splitlines()]
 assert samples[-1]['phase']=='after_exit' and samples[-1]['owned_group_processes']==[]
 strict=folder/'strict.private.json';journal=folder/'strict_trials.private.jsonl';index=folder/'executed_traces.private.json'
 assert sha(strict)==rec['private_strict_log_sha256'] and sha(journal)==rec['private_trial_journal_sha256'] and sha(index)==rec['private_executed_trace_index_sha256']
 motions=read(strict)['motions'];traces=read(index);assert len(motions)==len(traces)==1880
 trials={x['motion_id']:x['trials'][0] for x in motions};assert len(trials)==1880 and all(len(x['trials'])==1 for x in motions)
 expected=[dict(motion_id=k,trial=v) for k,v in trials.items()]
 encode=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
 assert sorted(map(encode,expected))==sorted(map(encode,[json.loads(x) for x in journal.read_text().splitlines()]))
 mode_stats={m:dict(trials=0,complete=0,failed=0,actual_control_steps=0,endpoint_distance_sum_m=0.,complete_endpoint_distance_sum_m=0.) for m in ['conditional_cfg5','unconditional']}
 seen=set();pairs={};byte_total=0;pose_samples=0
 for x in traces:
  d=owned(x['folder'],pathlib.Path('/home/ouyanghao'));assert d.parent==folder/'execution_private'
  m=read(d/'trace.private.json');binary=d/'qpos.private.bin';trial=m['trial']
  assert m['status']=='FINISHED_PHYSICAL_TRIAL' and m['runtime_kind']=='isaaclab_physics'
  assert m['motion_id'] not in seen and trials[m['motion_id']]==trial;seen.add(m['motion_id'])
  verify_trial(trial,trial['reference_frames'],0)
  assert m['samples']==x['samples']==trial['control_steps']+1
  assert binary.stat().st_size==m['binary_bytes']==m['samples']*36*4 and sha(binary)==m['binary_sha256']==x['binary_sha256']
  raw=binary.read_bytes();vals=[z[0] for z in struct.iter_unpack('<f',raw)];assert all(math.isfinite(v) for v in vals)
  for i in range(m['samples']):assert abs(sum(v*v for v in vals[i*36+3:i*36+7])-1)<.021
  byte_total+=len(raw);pose_samples+=m['samples']
  ep=x['final_reference_endpoint_position'];assert ep['compared_reference_frame']==trial['reference_frames']-1 and ep['full_T_minus_1_coverage']==trial['complete'] and ep['changes_strict_pass_rule'] is False
  assert math.isfinite(ep['mean_14_body_m']) and 0<=ep['mean_14_body_m']<=ep['max_14_body_m']
  v=mode_stats[x['mode']];v['trials']+=1;v['complete']+=int(trial['complete']);v['failed']+=int(trial['failed']);v['actual_control_steps']+=trial['control_steps']
  v['endpoint_distance_sum_m']+=ep['mean_14_body_m'];v['complete_endpoint_distance_sum_m']+=ep['mean_14_body_m']*int(trial['complete'])
  key=(x['segment_id'],x['seed']);assert x['mode'] not in pairs.setdefault(key,{});pairs[key][x['mode']]=trial['complete']
 assert seen==set(trials) and len(pairs)==940 and all(len(x)==2 for x in pairs.values())
 assert byte_total==rec['private_trace_bytes'] and rec['actual_scene_physics_steps']==rec['actual_vectorized_control_calls']*4
 assert sum(x['complete'] for x in mode_stats.values())==rec['complete_trials'] and sum(x['failed'] for x in mode_stats.values())==rec['failed_trials']
 for v in mode_stats.values():
  assert v['trials']==940 and v['complete']+v['failed']==940
  v['strict_complete_fraction']=v['complete']/940
  v['final_reference_mean_14_body_distance_all_trials_m']=v.pop('endpoint_distance_sum_m')/940
  v['final_reference_mean_14_body_distance_complete_trials_m']=v.pop('complete_endpoint_distance_sum_m')/v['complete'] if v['complete'] else None
 contingency={k:0 for k in ['both_complete','conditional_only_complete','unconditional_only_complete','neither_complete']}
 for v in pairs.values():
  c,u=v['conditional_cfg5'],v['unconditional'];key='both_complete' if c and u else 'conditional_only_complete' if c else 'unconditional_only_complete' if u else 'neither_complete';contingency[key]+=1
 fk=read(s['fk_receipt']);manifest=pathlib.Path(fk['private_manifest']['path']);assert sha(manifest)==fk['private_manifest']['sha256']
 for line in manifest.read_text().splitlines():
  x=json.loads(line);assert sha(x['file'])==x['sha256']
 audit=dict(all1880_strict_trials_and_terminal_poses_verified=True,private_journal_and_all_pose_bytes_sha_verified=True,
  source_fk_bytes_unchanged=True,finite_pose_samples=pose_samples,private_pose_bytes=byte_total,mode_statistics=mode_stats,paired_strict_contingency=contingency,
  original_strict_thresholds_unchanged=True,final_endpoint_is_separate_descriptive_distance=True,training_updates=0,new_model_loads=0,new_gpu_contexts=0,
  semantic_metrics_computed=False,full_validation_or_online_quality_accepted=False)
print(json.dumps(dict(run=run,observed_unix=time.time(),files=files,sha256=hashes,workers=workers,common_lock_free=free,closed_execution_verified=bool(closed),artifact_audit=audit,logs=logs,disks={str(p):shutil.disk_usage(p).free for p in [r,parent]})))
'''
j = json.loads(recorded_remote(code, out / (a.evidence + '_client.json'), timeout=240))
(out / (a.evidence + '.json')).write_text(json.dumps(j, indent=2), encoding='utf-8')
(out / 'progress.json').write_text(json.dumps(j, indent=2), encoding='utf-8')
print(json.dumps(j))
