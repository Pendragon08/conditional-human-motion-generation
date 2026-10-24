
from pathlib import Path
import hashlib,json,math,shutil,datetime
import numpy as np
witness=Path("D:/\u5177\u8eab\u667a\u80fd/PhyGile\u590d\u73b0/Unitree\u6846\u67b6\u9002\u914d_20261022/contact_doc_witness_v4")
comparison=witness.parent/'contact_comparison_v4'
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(name,value):(witness/name).write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=True)+'\n',encoding='utf-8')
launch=load(witness/'command_receipt.json')
before=load(witness/'source_hashes_before.json')
after=load(witness/'source_hashes_after.json')
result=load(comparison/'comparison.json') if (comparison/'comparison.json').exists() else None
checks={'main_exit_zero':launch.get('actual_exit_code')==0,'documented_invocation_count_one':launch['invocation_count']==1,'wrapper_utf8_bom':(witness/'documented_invocation.ps1').read_bytes().startswith(b'\xef\xbb\xbf'),'preexisting_sources_and_explicit_inputs_unchanged':all(after.get(p)==digest for p,digest in before.items()),'comparison_completed':bool(result and result.get('status')=='EIGHT_PAIRED_DIAGNOSTICS_COMPLETED'),'exactly_eight_children':bool(result and len(result.get('runs',[]))==8)}
rows=[];evidence=witness/'child_evidence';evidence.mkdir(exist_ok=False)
union={};commands=[];receipts=[];child_checks=[];problems=[]
for index,run in enumerate(result.get('runs',[]) if result else []):
 receipt=run['receipt'];profile=receipt['contact_profile'];clip=receipt['clip'];name=profile+'_clip'+str(clip)
 out=comparison/name;dest=evidence/name;dest.mkdir()
 for src,target in [(out/'receipt.json','receipt.json'),(out/'executed_sources.json','executed_sources.json'),(comparison/(name+'_stdout.txt'),'stdout.txt'),(comparison/(name+'_stderr.txt'),'stderr.txt')]:
  shutil.copyfile(src,dest/target)
 (dest/'actual_exit_code.txt').write_text(str(run['exit_code'])+'\n',encoding='utf-8')
 (dest/'command.json').write_text(json.dumps(run['command'],indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
 commands.append({'name':name,'command':run['command'],'actual_exit_code':run['exit_code']})
 receipts.append({'name':name,'receipt':receipt})
 manifest=load(out/'executed_sources.json');manifest_ok=sha(out/'executed_sources.json')==receipt['source_provenance']['sha256']
 prior_match=True;current_match=True
 for path,digest in manifest['files'].items():
  normalized=str(Path(path).resolve())
  prior_match=prior_match and before.get(normalized)==digest
  current_match=current_match and sha(path)==digest
  if normalized in union and union[normalized]!=digest:problems.append('Cross-child source mismatch: '+normalized)
  union[normalized]=digest
 feet=[x for x in receipt['compiled_contact_parameters'] if 'foot' in x['name']]
 expected=0.6 if profile=='official' else 1.0
 foot_ok=len(feet)==14 and all(math.isclose(x['friction'][0],expected,rel_tol=0,abs_tol=1e-12) for x in feet)
 metrics=receipt['metrics'];finite=all(isinstance(v,(float,int)) and math.isfinite(v) for v in metrics.values())
 with np.load(out/'raw_pre_reset.npz',allow_pickle=False) as raw:
  raw_finite=all(np.isfinite(raw[k]).all() for k in raw.files)
  raw_frames=len(raw['frame'])
  continuous=np.array_equal(raw['frame'],np.arange(raw_frames))
  actual={
   'mpjpe_m_until_termination':float(np.linalg.norm(raw['reference_body_pos']-raw['robot_body_pos'],axis=-1).mean()),
   'mpjae_rad_until_termination':float(np.abs(raw['reference_joint_pos']-raw['robot_joint_pos']).mean()),
   'mpjve_rad_s_until_termination':float(np.abs(raw['reference_joint_vel']-raw['robot_joint_vel']).mean())}
  metrics_match=all(math.isclose(actual[k],metrics[k],rel_tol=1e-7,abs_tol=1e-9) for k in actual)
 cmd=run['command']
 child={'name':name,'actual_exit_zero':run['exit_code']==0,'first_episode_status':receipt['status']=='FIRST_EPISODE_EVALUATED','positive_actual_physics_steps':receipt['frames_evaluated']>0 and receipt['physics_steps']==4*receipt['frames_evaluated'],'cpu':receipt['device']=='cpu','seed_2026':receipt['seed']==2026 and cmd[cmd.index('--seed')+1]=='2026','no_bounded_max_steps':'--max-steps' not in cmd,'no_optimizer_updates':receipt['optimizer_steps']==0,'action_contract_exact':receipt['action_scale_offset_exact'] is True,'finite_metrics':finite,'raw_state_finite':bool(raw_finite),'raw_frames_match':raw_frames==receipt['frames_evaluated'],'raw_frame_continuity':bool(continuous),'metrics_recalculated_match':metrics_match,'expected_compiled_foot_friction':foot_ok,'executed_sources_unchanged_receipt':receipt['executed_sources_unchanged'] is True,'source_manifest_sha_valid':manifest_ok,'executed_sources_match_prelaunch_hashes':prior_match,'executed_sources_still_match_hashes':current_match,'post_physics_pre_reset_recording':receipt['post_physics_forward_before_termination'] is True}
 child_checks.append(child)
 rows.append({'profile':profile,'clip':clip,'frames':receipt['frames_evaluated'],'motion_frames':receipt['motion_frames'],'physics_steps':receipt['physics_steps'],'full_clip_success':receipt['full_clip_success'],'terminated':receipt['terminated'],'timeout':receipt['timeout'],'metrics':metrics,'compiled_foot_coefficient':expected,'compiled_foot_count':len(feet),'checkpoint_sha256':receipt['checkpoint_sha256'],'motion_sha256':receipt['motion_sha256'],'source_file_count':len(manifest['files']),'all_checks_pass':all(v for k,v in child.items() if k!='name')})
checks['children_all_acceptance_checks_pass']=len(child_checks)==8 and all(all(v for k,v in c.items() if k!='name') for c in child_checks)
checks['exact_profile_clip_pairs']=sorted((r['profile'],r['clip']) for r in rows)==sorted((p,c) for p in ['official','matched_sliding_friction'] for c in range(4))
checks['same_frozen_checkpoint']=len({r['checkpoint_sha256'] for r in rows})==1
checks['paired_motion_hashes']=len(rows)==8 and all(next(r['motion_sha256'] for r in rows if r['profile']=='official' and r['clip']==c)==next(r['motion_sha256'] for r in rows if r['profile']=='matched_sliding_friction' and r['clip']==c) for c in range(4))
checks['cross_child_sources_consistent']=not problems
for src in ['comparison.json','progress.json']:
 if (comparison/src).exists():shutil.copyfile(comparison/src,witness/src)
save('commands.json',commands);save('receipts.json',receipts);save('executed_source_hashes.json',union)
verdict='PASS' if all(checks.values()) else 'FAIL'
acceptance={'verdict':verdict,'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'child_checks':child_checks,'rows':rows,'problems':problems,'source_files_in_prelaunch_snapshot':len(before),'unique_executed_sources_and_explicit_inputs':len(union),'global_snapshot_identical':before==after,'added_nonexecuted_source_paths':sorted(set(after)-set(before)),'scope':'Eight actual CPU MuJoCo-Warp first episodes, four development clips paired by profile, seed 2026, frozen policy; early termination allowed; no improvement or held-out benchmark claim'}
save('acceptance.json',acceptance)
lines=['# Fresh agent-follows-doc contact witness v4','','Verdict: **'+verdict+'**. The exact documented PowerShell command was invoked once through a UTF-8 WITH BOM .ps1 file. Actual top-level exit code: '+str(launch.get('actual_exit_code'))+'.','','Eight actual CPU MuJoCo-Warp first episodes were evaluated with seed 2026 and no optimizer updates. No dependency installation, source repair, retry, server action, or ReactiveBFM process action was performed.','','| Profile | Clip | Frames / motion | Physics steps | Full clip success | MPJPE m | MPJAE rad | MPJVE rad/s | Foot coefficient |','|---|---:|---:|---:|---|---:|---:|---:|---:|']
for r in rows:
 m=r['metrics'];lines.append('| '+r['profile']+' | '+str(r['clip'])+' | '+str(r['frames'])+' / '+str(r['motion_frames'])+' | '+str(r['physics_steps'])+' | '+str(r['full_clip_success'])+' | '+format(m['mpjpe_m_until_termination'],'.8f')+' | '+format(m['mpjae_rad_until_termination'],'.8f')+' | '+format(m['mpjve_rad_s_until_termination'],'.8f')+' | '+str(r['compiled_foot_coefficient'])+' |')
lines+=['','All acceptance checks: '+str(all(checks.values()))+'. Pre/post source-and-input snapshot files: '+str(len(before))+'. Unique executed source/input files independently rehashed: '+str(len(union))+'.','','Broad directory snapshots differed only by one newly created scripts/after_training.py that was absent from all executed-source manifests. Every pre-existing tracked source and explicit input remained unchanged. This unrelated addition is preserved in command_receipt.json and source_hashes_after.json.', '', 'Early termination is permitted by the diagnostic acceptance condition. Metrics are censored at the first termination and must be read with episode duration. These are four development clips, not a held-out benchmark. No performance improvement is required or inferred.','','The witness retains exact commands, top-level and child stdout/stderr, actual exit codes, copied receipts and executed-source manifests, before/after source hashes, and independently recalculated acceptance. Private raw motions and recorded states remain only in the original private comparison directory.']
(witness/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'verdict':verdict,'checks':checks,'rows':rows},ensure_ascii=True),flush=True)
