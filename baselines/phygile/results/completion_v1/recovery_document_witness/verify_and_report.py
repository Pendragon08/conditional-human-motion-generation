from pathlib import Path
import datetime, hashlib, json, re
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RAW=OUT/'raw_result'
REMOTE_PREFIX='results/completion_v1/closure_recovery/'
checks=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def npz(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def check(name,passed,details=None):checks.append({'name':name,'passed':bool(passed),'details':details})
def delta(a,b):return float(np.max(np.abs(np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))))

manifest=read(OUT/'remote_manifest.stdout.log')
integrity=[]
for item in manifest['files']:
 p=RAW/item['path'].removeprefix(REMOTE_PREFIX)
 integrity.append({**item,'copy_sha256':sha(p),'copy_bytes':p.stat().st_size})
check('remote_raw_manifest_matches_local_copy',all(x['sha256']==x['copy_sha256'] and x['bytes']==x['copy_bytes'] for x in integrity),{'file_count':len(integrity),'total_bytes':sum(x['bytes'] for x in integrity)})
(OUT/'verified_raw_manifest.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')

run=read(OUT/'run.result.json')
receipt=read(RAW/'run/receipt.json')
outer=read(RAW/'verification.json')
primary=read(RAW/'output/verification.json')
sentinel=(OUT/'run.stdout.log').read_text(encoding='utf-8',errors='replace').count('PHYGILE_TERMINAL_BOUNDARY_COMPLETE')
check('documented_invocation_once_exit_zero',run['exit_code']==0 and sentinel==1,{'invocations':1,'exit_code':run['exit_code'],'sentinel_count':sentinel,'elapsed_seconds':receipt['elapsed_seconds']})
check('inner_recorded_command_completed',receipt['status']=='COMPLETED' and receipt['exit_code']==0 and not receipt.get('timed_out',False),{'status':receipt['status'],'exit_code':receipt['exit_code'],'timeout_seconds':receipt['timeout_seconds']})
check('outer_verification_authenticates_primary',outer['status']=='COMPLETED' and outer['primary_sha256']==sha(RAW/'output/verification.json'),outer)
check('primary_engineering_validation_completed',primary['status']=='COMPLETED_ENGINEERING_VALIDATION',primary['status'])

doc_sha=sha(OUT/'invocation_document.snapshot.md')
check('invocation_doc_unchanged_and_remote_same',doc_sha==sha(ROOT/'docs/terminal_recovery_invocation.md')==manifest['inputs']['docs/terminal_recovery_invocation.md']['sha256'],{'sha256':doc_sha})
spec=json.loads(manifest['inputs']['configs/env-spec.json']['text'])
spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
ledger=manifest['inputs']['docs/environment.md']['text']
check('environment_spec_matches_existing_server_notes_ledger',f'`{spec_hash}`' in ledger,{'spec_hash':spec_hash,'ledger':'docs/environment.md','ledger_sha256':manifest['inputs']['docs/environment.md']['sha256'],'spec_sha256':manifest['inputs']['configs/env-spec.json']['sha256']})
sources=[{'path':n,'run_sha256':h,'local_sha256':sha(ROOT/n) if (ROOT/n).is_file() else None} for n,h in primary['source_sha256'].items()]
check('recorded_implementation_sources_match_read_only_local_sources',all(x['run_sha256']==x['local_sha256'] for x in sources),{'files':len(sources),'mismatches':[x for x in sources if x['run_sha256']!=x['local_sha256']]})

workers=sorted((RAW/'output/simulator').glob('invocation_*'))
worker_records=[read(p/'worker_result.json') for p in workers]
check('all_isolated_workers_completed_with_frozen_gmt',len(workers)==7 and all(x['status']=='COMPLETED' and x['gmt_parameters_unchanged'] and x['checkpoint_sha256']==primary['checkpoint_sha256'] for x in worker_records),{'workers':len(workers),'gmt_parameter_digests':sorted(set(x['gmt_parameter_digest'] for x in worker_records))})
check('worker_request_and_physical_file_hashes_match',all(x['request_sha256']==sha(p/'request.json') and x['determinism']['physical_properties_sha256']==sha(p/'initial_physical_properties.npz') for p,x in zip(workers,worker_records)))
physical0=npz(workers[0]/'initial_physical_properties.npz')
physical=[]
for p in workers:
 values=npz(p/'initial_physical_properties.npz')
 physical.append({'worker':p.name,'arrays':{k:{'shape':list(values[k].shape),'max_abs_delta_from_initialize':delta(values[k],physical0[k]),'exact_equal':bool(np.array_equal(values[k],physical0[k]))} for k in physical0}})
check('all_initial_physical_parameters_exactly_equal',all(v['exact_equal'] for x in physical for v in x['arrays'].values()),physical)
check('documented_deterministic_worker_flags',all(x['determinism']['torch_deterministic_algorithms'] and x['determinism']['cublas_workspace_config']==':4096:8' and x['determinism']['tf32'] is False and x['determinism']['physx_enhanced_determinism'] for x in worker_records))

first=npz(workers[1]/'raw.npz'); second=npz(workers[2]/'raw.npz')
rollback={k:{'max_abs_delta':delta(first[k],second[k]),'exact_equal':bool(np.array_equal(first[k],second[k])),'shape':list(first[k].shape)} for k in first}
check('independent_ten_frame_replays_exactly_equal',all(x['exact_equal'] for x in rollback.values()) and first['states'].shape[0]==10,rollback)
check('raw_replay_matches_reported_state_and_body_differences',primary['rollback_witness']['state_max_abs_delta']==rollback['states']['max_abs_delta'] and primary['rollback_witness']['body_max_abs_delta_m']==rollback['body_positions']['max_abs_delta'] and max(rollback['states']['max_abs_delta'],rollback['body_positions']['max_abs_delta'])<=5e-4)

fixture=primary['actual_feedback_fixture']
raw3=npz(workers[3]/'raw.npz');raw4=npz(workers[4]/'raw.npz')
check('two_one_frame_feedback_proxy_acceptances',fixture['evaluation_type']=='self_supervised_proxy' and fixture['accepted_windows']==2 and fixture['control_frames']==2 and fixture['window_frames']==1 and all(x['accepted'] and x['recorded_frames']==1 for x in fixture['decisions']) and fixture['input_prefix_sha256'][0]!=fixture['input_prefix_sha256'][1],fixture)
check('accepted_history_replay_verified_before_continuation',len(worker_records[4]['replayed_windows'])==1 and worker_records[4]['replayed_windows'][0]['max_state_difference']==0 and np.array_equal(raw3['states'][-1],raw4['actual_start_state']),{'history_proof':worker_records[4]['replayed_windows'],'next_start_state_delta':delta(raw3['states'][-1],raw4['actual_start_state'])})

target=npz(RAW/'output/nominal_standing_target.npz')
target_state=target['states'][0]
initial=npz(workers[0]/'current.npz')['states'][0]
config=json.loads(manifest['inputs']['configs/nominal_standing_reference.json']['text'])
skeleton=read(ROOT/'configs/g1_codec_skeleton.json')
expected_q=np.full(29,float(config['unlisted_joint_position']))
for pattern,value in config['joint_positions'].items():
 for i,n in enumerate(skeleton['joint_names']):
  if re.fullmatch(pattern,n):expected_q[i]=float(value)
w,x,y,z=initial[3:7]/np.linalg.norm(initial[3:7])
yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
expected_quat=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
if expected_quat[0]<0:expected_quat=-expected_quat
check('nominal_standing_state_independently_matches_config',np.array_equal(target_state[13:42],expected_q) and target_state[2]==config['root_height_m'] and np.array_equal(target_state[:2],initial[:2]) and delta(target_state[3:7],expected_quat)<1e-14 and np.all(target_state[7:13]==0) and np.all(target_state[42:]==0),{'root_height_m':float(target_state[2]),'root_xy_matches_initial':bool(np.array_equal(target_state[:2],initial[:2])),'orientation_delta_from_initial_yaw_only':delta(target_state[3:7],expected_quat),'joint_position_delta':delta(target_state[13:42],expected_q),'root_velocity_max_abs':float(np.max(np.abs(target_state[7:13]))),'joint_velocity_max_abs':float(np.max(np.abs(target_state[42:]))),'source':config['source'],'author_suffix_verified':config['author_suffix_verified']})
proposals=[]
for p in sorted((RAW/'output/proposals').glob('*.npz')):
 d=npz(p)
 proposals.append({'path':str(p.relative_to(RAW)),'new_frames':len(d['candidate_native'])-len(d['conditioned_actual_prefix']),'terminal_native_exact_equal':bool(np.array_equal(d['candidate_native'][-1:],target['states'])),'terminal_body_positions_exact_equal':bool(np.array_equal(d['candidate_body_positions'][-1:],target['body_positions'])),'native_max_abs_delta':delta(d['candidate_native'][-1:],target['states']),'body_max_abs_delta':delta(d['candidate_body_positions'][-1:],target['body_positions'])})
check('every_fifty_frame_candidate_ends_at_independent_nominal_target',len(proposals)==2 and all(x['new_frames']==50 and x['terminal_native_exact_equal'] and x['terminal_body_positions_exact_equal'] for x in proposals),proposals)
check('rejected_generation_has_no_invalid_commit',primary['outcome']=='REJECTED_WITHOUT_INVALID_COMMIT' and primary['accepted_attempts']==0 and primary['rejected_attempts']==2 and primary['committed_new_frames']==0 and primary['initial_prefix_sha256']==primary['committed_prefix_sha256'],{'outcome':primary['outcome'],'accepted_attempts':primary['accepted_attempts'],'rejected_attempts':primary['rejected_attempts'],'committed_new_frames':primary['committed_new_frames'],'attempts':primary['attempts']})
check('scope_is_untrained_generator_and_zero_ppo_updates',primary['trained_generator'] is False and primary['trained_semantic_generation_claim'] is False and primary['physics_prefix_ppo_adaptation_updates']==0 and primary['gmt_parameters_unchanged'] and primary['generator_parameters_unchanged'])

report={'status':'PASS_ENGINEERING_DOCUMENT_WITNESS' if all(x['passed'] for x in checks) else 'FAIL_DOCUMENT_WITNESS','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independence':{'only_invocation_doc_and_implementation_read':True,'main_task_history_read':False,'implementation_or_threshold_or_document_edited':False,'rerun_count':0,'documented_command_invocations':1,'ledger_path_supplied_by_parent_after_start':'docs/environment.md; read and independently hash checked'},'document_sha256':doc_sha,'remote_host':'ouyanghao@10.147.17.32:22','remote_cwd':manifest['remote_cwd'],'documented_command':'bash scripts/run_isaac.sh scripts/validate_terminal_boundary.py --name closure_recovery','preflight_stdout':(OUT/'preflight.stdout.log').read_text(),'run':run,'checks':checks,'document_reality_divergences':[],'scope_limits':['Both untrained diffusion proposals were rejected; zero generated frames were committed.','Nominal zero-velocity standing is an explicit target, not demonstrated stable standing.','Two accepted one-frame held-pose windows are a feedback-interface proxy, not a performance benchmark.','No trained generator, semantic generation, closed-loop PPO adaptation, or paired dataset claim.'],'source_comparisons':sources,'raw_manifest':'verified_raw_manifest.json','raw_copy':'raw_result','primary_verification_sha256':sha(RAW/'output/verification.json'),'failed_checks':[x['name'] for x in checks if not x['passed']]}
(OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
md=f'''# Independent recovery document witness

Status: **{report['status']}**. Ran the documented command exactly once, with no implementation, threshold, or document edits and no retry. Main task history was not read.

Command: `{report['documented_command']}`

Server: `{report['remote_host']}`; directory: `{report['remote_cwd']}`. Preflight found GPU0 RTX A6000 at 2968 / 49140 MiB, no task process and no pre-existing closure_recovery directory. Existing idle allocation was reused as documented.

Outer SSH exit: {run['exit_code']}; recorded child exit: {receipt['exit_code']}; timeout: {receipt['timeout_seconds']} s; elapsed: {receipt['elapsed_seconds']:.2f} s. Completion sentinel count: {sentinel}. Both verification JSON files exist and their hash linkage matches.

The ten-frame independent replays have exactly equal state, body position, action, termination and start-state arrays (maximum differences all 0). All six initial physical-property arrays are exactly equal across all seven isolated workers. Accepted-history replay reports 0 difference, and raw next-start state equals the prior accepted continuation.

Both 50-new-frame candidates end exactly at the separately saved nominal standing target in native state and body positions. Independent state checks confirm configured joint positions, 0.76 m root height, initial XY/yaw, zero pitch/roll, and zero root/joint velocities.

Both untrained diffusion candidates were rejected; accepted generated attempts = 0, rejected = 2, committed new frames = 0. The original prefix remained committed. Two accepted one-frame held-pose windows only witness the acceptance/feedback interface. This does not demonstrate generated-motion success or stable standing. No closed-loop PPO updates occurred.

Canonical environment spec hash `{spec_hash}` matches the existing `docs/environment.md` server-notes ledger. No doc-vs-reality divergence was found in the invocation or stated scope. The parent supplied the ledger/spec paths after launch; this timing is recorded in JSON.

Invocation document SHA256: `{doc_sha}`.
Primary verification SHA256: `{report['primary_verification_sha256']}`.
Preserved {len(integrity)} raw files ({sum(x['bytes'] for x in integrity)} bytes), all verified against remote SHA256. See `verified_raw_manifest.json`, `remote_manifest.stdout.log`, `raw_result/`, `run.command.json`, `run.stdout.log`, `run.stderr.log`, and `run.result.json`.

Failed checks: {report['failed_checks']}.
'''
(OUT/'report.md').write_text(md,encoding='utf-8')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'doc_sha256':doc_sha,'raw_files':len(integrity),'raw_bytes':sum(x['bytes'] for x in integrity),'primary_sha256':report['primary_verification_sha256']},indent=2))
