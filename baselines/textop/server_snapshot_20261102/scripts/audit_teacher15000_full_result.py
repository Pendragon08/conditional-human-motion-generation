"""CPU-only acceptance of a CLOSED actual all-source replay; never simulates.

All original decisions remain private. A selected motion is an actual complete
strict rollout, not a claim about repeated-trial failure probability or the
paper's historical EMA candidate gate. No Torch, SDK or model import is used.
"""
import argparse,array,hashlib,json,math,os,pathlib,sys,time,traceback
from strict_tracking_coverage import verify_trial
from teacher15000_strict_protocol import STATUS,QUALITY,load_teacher,settings_signature

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN='teacher15000_direct_full_v1'
CHECKPOINT='66e87c9471d2860f1b43a18a41e938c12647c93212f1fe44c8be1f1b27febc88'
STRICT_SPEC='91184fff75f8a0579a759abb418d4a0186d3e42a9a48c176e3d33facc768a1cd'

def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))

def owned(path,parent):
 path=pathlib.Path(path);parent=pathlib.Path(parent)
 assert path==path.resolve() and parent==parent.resolve() and (path==parent or parent in path.parents)
 here=path
 while True:
  assert here.stat().st_uid==1013,(str(here),'owner')
  if here==parent:break
  here=here.parent
 return path

def validate_pose_bytes(raw,samples):
 assert sys.byteorder=='little' and len(raw)==samples*36*4
 values=array.array('f');values.frombytes(raw)
 assert all(math.isfinite(v) for v in values),'Nonfinite actual pose'
 assert all(.99<math.sqrt(sum(x*x for x in values[k+3:k+7]))<1.01 for k in range(0,len(values),36))

def load_closed_run():
 sp=ROOT/'configs/teacher15000-direct-strict-spec.json';assert sha(sp)==STRICT_SPEC
 spec=read(sp);assert spec['full_launch_enabled'] and spec['strict_rollouts_per_motion']==1
 load_teacher(ROOT,spec)
 for n,k in [('evaluate_teacher15000_strict.py','executor_sha256'),('teacher15000_strict_protocol.py','protocol_sha256'),
             ('direct_strict_trace.py','trace_sha256'),('supervise_teacher15000_strict.py','supervisor_sha256')]:assert sha(ROOT/'scripts'/n)==spec[k]
 out=ROOT/'tracker_validation'/RUN;prefix='teacher15000_strict_'+RUN
 paths={'receipt':out/'receipt.json','outer':ROOT/'receipts'/(prefix+'_outer_exit.json'),
  'exit':ROOT/'receipts'/(prefix+'_exit.json'),'process':ROOT/'receipts'/(prefix+'_process_exit.json'),
  'resources':ROOT/'logs'/(prefix+'_resources.jsonl')}
 assert all(p.is_file() for p in paths.values()),'Real full replay has not closed; no audit input yet'
 for p in paths.values():owned(p,ROOT)
 result={k:read(p) for k,p in paths.items() if k!='resources'}
 r,e,pr,outer=[result[k] for k in ['receipt','exit','process','outer']]
 assert r['status']==STATUS and r['quality_label']==QUALITY and r['stage']=='full' and r['mode']=='teacher15000_strict'
 assert r['output_writers_closed'] and r['physical_filter_executed'] and not r['dataset_accepted']
 assert r['optimizer_updates']==0 and r['policy_checkpoint_completed_iterations']==15000 and r['policy_checkpoint_optimizer_updates']==300000
 assert r['policy_checkpoint_sha256']==CHECKPOINT and r['pilot_spec_sha256']==STRICT_SPEC
 assert r['execution_settings_sha256']==settings_signature(spec)
 assert r['source_sha256']==spec['executor_sha256'] and r['evaluation_protocol_sha256']==spec['protocol_sha256'] and r['trace_source_sha256']==spec['trace_sha256']
 assert r['reference_sources']==r['replayed_original_sources']==r['finished_trials']==8278
 assert r['complete_trials']+r['failed_trials']==8278 and r['actual_scene_physics_steps']==4*r['actual_vectorized_control_calls']
 assert r['terminal_execution_state_captured_before_autoreset'] and r['direct_replay_bypasses_historical_ema_selection']
 assert outer['actual_returncode']==0
 for x in [e,pr]:
  assert x['child_reaped'] and x['child_exit_code']==0 and x['supervision_passed'] and not x['timed_out']
  assert x['supervisor_error'] is None and not x.get('cleanup_error') and x['sample_errors']==[]
 assert e['evaluation_receipt_sha256']==sha(paths['receipt']) and e['process_exit_sha256']==sha(paths['process'])
 assert e['resources_sha256']==sha(paths['resources'])
 resources=[json.loads(x) for x in paths['resources'].read_text().splitlines()]
 assert len(resources)==e['samples_written']==pr['samples_written'] and resources[0]['phase']=='before_launch' and resources[-1]['phase']=='after_exit'
 assert all(x['available_host_ram_bytes']>=2*1024**3 and x['data_free_bytes']>=2*1024**3 for x in resources)
 # The original process group must be gone. Never signal it from this audit.
 group=pr['child_pid']
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid==1013:assert os.getpgid(int(p.name))!=group,'Original replay process group remains alive'
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 return spec,r,out,{k:sha(p) for k,p in paths.items()}

def audit(output):
 assert os.getuid()==1013 and set(os.sched_getaffinity(0))=={0,1,2,3}
 spec,r,run_dir,hashes=load_closed_run()
 plan=read(ROOT/'configs/filter-training-plan.json');rows=[];manifest_hashes={}
 parent=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')
 for name in plan['input_reference_exports']:
  folder=owned(parent/name,parent);mp=owned(folder/'manifest.private.jsonl',parent);meta=read(folder/'receipt.json')
  assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED' and sha(mp)==meta['private_manifest_sha256']
  manifest_hashes[str(mp)]=sha(mp)
  rows.extend(x for x in map(json.loads,mp.read_text().splitlines()) if not x['mirrored'])
 assert len(rows)==len({x['canonical_source'] for x in rows})==8278
 # Same canonical metadata formula as filter_training_runtime.fingerprint;
 # importing that training module would load Torch/SDK, which this audit avoids.
 fingerprint=hashlib.sha256(json.dumps([{'sha256':x['sha256'],'frames':x['frames']} for x in rows],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 assert fingerprint==r['reference_fingerprint']==spec['expected_reference_fingerprint']
 ids={hashlib.sha256(x['canonical_source'].encode()).hexdigest():i for i,x in enumerate(rows)};assert len(ids)==8278
 strict=run_dir/'strict.private.json';journal=run_dir/'strict_trials.private.jsonl';index=run_dir/'executed_traces.private.json'
 for p in [strict,journal,index]:owned(p,ROOT)
 assert sha(strict)==r['private_strict_log_sha256'] and sha(journal)==r['private_trial_journal_sha256'] and sha(index)==r['private_executed_trace_index_sha256']
 decisions=read(strict);motions=decisions['motions'];traces=read(index)
 assert decisions['status']=='DIRECT_STRICT_TRIALS_FINISHED' and not decisions['diagnostic_only'] and decisions['quality_label']==QUALITY
 assert decisions['policy_checkpoint_sha256']==CHECKPOINT and decisions['reference_fingerprint']==r['reference_fingerprint']
 assert decisions['contact_profile_sha256']==r['contact_profile_sha256']
 assert len(motions)==len(traces)==8278 and {x['motion_id'] for x in motions}==set(ids)
 trial_by_id={};canonical=[];selected=[];failed=0
 for item in motions:
  mid=item['motion_id'];row=rows[ids[mid]];assert len(item['trials'])==1
  t=item['trials'][0];verify_trial(t,row['frames'],0);trial_by_id[mid]=t
  path=owned(pathlib.Path(row['file']),parent);assert sha(path)==row['sha256']
  entry=dict(global_index=ids[mid],canonical_source=row['canonical_source'],split=row['split'],frames=row['frames'],
   reference_file=str(path),reference_sha256=row['sha256'],motion_id=mid,trial=t,strict_complete_pass=bool(t['complete'] and not t['failed']))
  canonical.append(entry)
  if entry['strict_complete_pass']:selected.append(entry)
  else:failed+=1
 expected=[dict(motion_id=x['motion_id'],trial=x['trials'][0]) for x in motions]
 encode=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'))
 assert sorted(map(encode,expected))==sorted(map(encode,map(json.loads,journal.read_text().splitlines())))
 trace_ids=set();total=0;pose_samples=0;joint_names=None
 for trace in traces:
  folder=owned(pathlib.Path(trace['folder']),run_dir/'execution_private')
  assert folder.parent==run_dir/'execution_private'
  meta=read(owned(folder/'trace.private.json',ROOT));bp=owned(folder/'qpos.private.bin',ROOT);mid=meta['motion_id']
  assert mid in ids and mid not in trace_ids;trace_ids.add(mid);t=trial_by_id[mid]
  assert meta['schema']=='textop.compact_physics_trace.v1' and meta['status']=='FINISHED_PHYSICAL_TRIAL' and meta['runtime_kind']=='isaaclab_physics'
  assert meta['trial']==t and meta['reference_frames']==rows[ids[mid]]['frames'] and meta['samples']==trace['samples']==t['control_steps']+1
  assert meta['row_width']==36 and meta['dtype']=='little_endian_float32' and meta['control_hz']==50
  if joint_names is None:joint_names=meta['joint_names'];assert len(set(joint_names))==29
  assert meta['joint_names']==joint_names
  raw=bp.read_bytes();validate_pose_bytes(raw,meta['samples'])
  assert len(raw)==meta['binary_bytes']==trace['binary_bytes'] and hashlib.sha256(raw).hexdigest()==meta['binary_sha256']==trace['binary_sha256']
  assert trace['physical_failure']==t['failed'] and all(math.isfinite(v) for k,v in meta['metrics'].items() if isinstance(v,(int,float)))
  total+=len(raw);pose_samples+=meta['samples']
 assert trace_ids==set(ids) and total==r['private_trace_bytes'] and total<=spec['maximum_private_trace_bytes']
 assert len(selected)==r['complete_trials'] and failed==r['failed_trials']
 canonical.sort(key=lambda x:x['global_index']);selected.sort(key=lambda x:x['global_index'])
 output=pathlib.Path(output);assert output.parent==ROOT/'filtered15000_20261027';owned(output.parent,ROOT);output.mkdir(mode=0o700,exist_ok=False)
 artifacts={}
 for name,payload in [('all_decisions.private.json',canonical),('accepted_sources.private.json',selected)]:
  p=output/name
  with p.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  artifacts[name]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
 return dict(schema='textop.closed_direct_strict15000_audit.v1',status='CLOSED_FULL_STRICT_RESULT_CPU_AUDIT_PASSED',
  source_count=8278,attempted=8278,complete_passes=len(selected),failed=failed,unattempted=0,
  reference_fingerprint=r['reference_fingerprint'],checkpoint_sha256=CHECKPOINT,strict_spec_sha256=STRICT_SPEC,
  raw_execution_receipt_sha256=hashes['receipt'],input_evidence_sha256=hashes,private_artifacts=artifacts,
  prepared_reference_manifest_sha256=manifest_hashes,trace_bytes_finite_verified=total,pose_samples_verified=pose_samples,
  original_processes_exited=True,strict_result_accepted=True,filtered_corpus_accepted=False,generator_started=False,
  historical_ema_candidates=0,ema_gate_bypassed_by_user=True,paper_two_stage_route_reproduced=False,
  strict_repetitions_per_source=1,final_reference_endpoint_comparison_performed=False,semantic_quality_accepted=False,
  new_model_loads=0,new_gpu_contexts=0,new_sdk_runs=0,new_physics_steps=0,optimizer_updates=0,output_writers_closed=True,
  source_sha256=sha(__file__))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();started=time.monotonic()
 try:
  result=audit(a.output);result['elapsed_seconds']=time.monotonic()-started
  dest=pathlib.Path(a.output)/'audit.json'
  with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
  print(json.dumps(result),flush=True)
 except BaseException:
  traceback.print_exc();raise
