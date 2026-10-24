from pathlib import Path
import json, hashlib, re
import numpy as np
w = Path(r'D:\具身智能\PhyGile复现\Unitree框架适配_20261022\closure_mjlab_doc_witness_v1')
o = w.parent / 'closure_mjlab_v1'
a = w.parent / 'private_assets_contact_v3'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
v=read(o/'verification.json')
r=read(w/'process_result.json')
s=read(w/'source_integrity.json')
provenance=read(o/'executed_sources.json')
source_mismatches=[str(p) for p,h in provenance['files'].items() if not Path(p).is_file() or sha(p)!=h]
raw_mismatches=[p for p,h in v['raw_files'].items() if not (o/p).is_file() or sha(o/p)!=h]
shapes={}
for p in [o/'committed_prefix.npz', *sorted((o/'proposals').glob('*.npz'))]:
    with np.load(p,allow_pickle=False) as arrays:
        shapes[str(p.relative_to(o))]={k:{'shape':list(arrays[k].shape),'dtype':str(arrays[k].dtype)} for k in arrays.files}
env=read(w/'environment_plan_v2.json')
env_hash=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')).hexdigest()
command=re.search(r'(?ms)~~~powershell\r?\n(.*?)\r?\n~~~',(w/'MJLAB_CLOSURE_V1_INVOCATION.md').read_text(encoding='utf-8-sig')).group(1)
checks={
'exact_command_matches_document':(w/'command.txt').read_text(encoding='utf-8-sig')==command,
'powershell_wrapper_utf8_bom':(w/'run_documented_once.ps1').read_bytes().startswith(b'\xef\xbb\xbf'),
'single_invocation_actual_exit_zero':r['invocation_count']==1 and r['actual_exit_code']==0,
'within_20_minute_limit':r['duration_seconds']<=1200,
'environment_spec_hash_matches_ledger':env_hash=='a43f960bd3bcdcd3398655bffe291138b73de5ca7f30901765172ff9821e1954',
'completed_engineering_validation':v['status']=='COMPLETED_ENGINEERING_VALIDATION',
'actual_mujoco_warp_and_diffusion':v['actual_mujoco_warp_physics'] and v['actual_diffusion_sampling'],
'checkpoint_unchanged_and_hash_matches':v['checkpoint_unchanged'] and sha(a/'parent_checkpoint.pt')==v['checkpoint_sha256'],
'normalization_hash_matches':sha(a/'normalization.npz')==v['generator']['normalization_sha256'],
'generator_parameters_unchanged':v['generator_parameters_unchanged'],
'snapshot_29_sources_unchanged':s['file_count']==29 and s['before_all_match'] and s['after_all_match'] and s['manifest_before_sha256']==s['manifest_after_sha256'],
'executed_source_manifest_hash_matches':sha(o/'executed_sources.json')==v['source_provenance']['sha256'],
'executed_255_sources_still_match':len(provenance['files'])==255 and not source_mismatches and v['executed_sources_unchanged'],
'fresh_scene_replay_delta_within_2e_6':v['rollback_witness']['state_max_abs_delta']<=2e-6 and v['rollback_witness']['body_max_abs_delta_m']<=2e-6,
'accepted_history_replay_exact':all(x['max_state_difference']==0 for scene in v['simulator_invocations'] for x in scene['replayed_windows']) and sum(len(scene['replayed_windows']) for scene in v['simulator_invocations'])>=1,
'all_scenes_fresh':all(scene['fresh_scene'] for scene in v['simulator_invocations']),
'held_initialization_proxy_two_accepted_one_frame_windows':v['actual_feedback_fixture']['evaluation_type']=='self_supervised_proxy' and v['actual_feedback_fixture']['accepted_windows']==2 and v['actual_feedback_fixture']['window_frames']==1 and len(v['actual_feedback_fixture']['decisions'])==2 and all(x['accepted'] and x['expected_frames']==1 and x['recorded_frames']==1 for x in v['actual_feedback_fixture']['decisions']),
'exactly_two_50_frame_diffusion_attempts':len(v['attempts'])==2 and len(v['proposals'])==2 and all(x['expected_frames']==50 for x in v['attempts']) and all(x['new_frames']==50 for x in v['proposals']),
'failed_motions_never_committed':all(not x['accepted'] for x in v['attempts']) and v['committed_new_frames']==0 and v['outcome']=='REJECTED_WITHOUT_INVALID_COMMIT',
'raw_array_hashes_match':not raw_mismatches,
'no_trained_or_semantic_generation_claim':not v['trained_generator'] and not v['trained_semantic_generation_claim'] and not v['generator']['trained'] and v['optimizer_updates']==0
}
result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'environment_canonical_sha256':env_hash,'source_mismatches':source_mismatches,'raw_mismatches':raw_mismatches,'array_metadata_only':shapes,'actual_process_result':r,'doc_reality_divergences':[],'nonfatal_stderr':'Torch jit.script FutureWarning; PowerShell 5.1 records native stderr as NativeCommandError but actual Python exit is 0','scientific_scope':'Untrained unconditional 262D diffusion wiring and real simulation feedback/rejection/replay only; two accepted one-frame proxies are held-initialization references, not learned motion or dataset truth.'}
(w/'independent_artifact_verification.json').write_text(json.dumps(result,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=True,indent=2))