"""Independent read-only audit of the single documented V27 invocation."""
import sys
sys.path.insert(0, '/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages')
import json, hashlib, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT=ROOT/'results/fixed_reference_com_v27_20261012'
V24=ROOT/'results/contact_com_validation_v24_20261012'
BASE=V24/'raw/val_planned_s11'
RAW=OUT/'raw/val_raw_replay_s11'
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def object_hash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def motion_hash(p):
    h=hashlib.sha256()
    with np.load(p,allow_pickle=False) as a:
        for k in sorted(a.files):
            x=np.ascontiguousarray(a[k]);h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()
checks={}
def check(k,v): checks[k]=bool(v)
version=read(OUT/'versions.json');meta=read(RAW/'metadata.json');base_meta=read(BASE/'metadata.json')
rows=read(RAW/'metrics.json');base_rows=read(BASE/'metrics.json');receipt=read(RAW/'receipt.json')
selection=read(OUT/'selected/raw_reference.json');line=read(RAW/'lineage.json')
req=read(OUT/'requests/val_raw_replay_s11.json');config=read(ROOT/'configs/physics_guided_20261011.json')
v26=read(ROOT/'results/contact_com_runtime_v26_20261012/run_receipt.json');v24=read(V24/'validation_summary.json')
file_bindings={p:{'expected':s,'actual':sha(ROOT/p)} for p,s in version['files'].items()}
check('all_25_source_document_config_version_file_bindings',all(x['expected']==x['actual'] for x in file_bindings.values()))
check('v26_pass_and_v24_432_attempts',v26['status']=='PASS' and v24['attempts']==432 and sum(x['inputs'] for x in v24['entries'])==432)
check('v26_outputs_preserved',all(sha(V24/p)==s for p,s in v26['preserved_outputs'].items()))
check('source_manifest_and_thresholds_unchanged',all(sha(OUT/p)==sha(V24/p) for p in ('source_manifest.json','frozen_thresholds.json')))
check('v24_parent_bindings',all(sha(BASE/p)==s for p,s in version['parent_outputs'].items()))
check('execution_request_bound',req==line['request'] and line['certification']=='bound_request_at_execution')
for key,p in [('config',ROOT/'configs/physics_guided_20261011.json'),('manifest',OUT/'source_manifest.json'),('thresholds',OUT/'frozen_thresholds.json'),('versions',OUT/'versions.json')]:
    check('request_'+key+'_sha',req[key]==sha(p))
check('request_replay_motion_sha',req['replay_motion']==motion_hash(OUT/'selected/raw_reference.npz'))
check('outputs_bound_at_execution',all(sha(RAW/p)==s for p,s in line['outputs'].items()))
check('receipt_72_first_attempt',receipt['status']=='complete' and receipt['cases']==len(meta['cases'])==len(rows)==72 and receipt['trajectories_sha256']==sha(RAW/'trajectories.npz'))
check('24_sources_three_offsets_same_order',meta['cases']==base_meta['cases'] and len({r['clip_id'] for r in meta['cases']})==24 and len({r['offset_m'] for r in meta['cases']})==3)
check('single_original_candidate_all_72_cases',selection['construction_candidates']==1 and len(selection['choices'])==72 and all(c['candidate']==0 and c['path']==str(BASE) and c['case_id']==r['case_id'] and c['metrics']==r and c['construction_complete']==r['completed'] for c,r in zip(selection['choices'],base_rows)))
check('original_goals_unchanged',meta['original_goal']==base_meta['original_goal'] and meta['original_goal_sha256']==base_meta['original_goal_sha256']==selection['original_goal_sha256'] and meta['proposed_goal']==meta['original_goal'] and meta['proposed_heading']==meta['original_heading'])
check('seven_initial_state_differences_zero',len(meta['manual_reset_vs_first_action_state_max_difference'])==7 and all(x==0 for x in meta['manual_reset_vs_first_action_state_max_difference'].values()) and meta['initial_reset_gate_pass'])
check('no_diffusion_calls_or_construction_batches',meta['generation_batch_calls']==meta['upstream_warmup_diffusion_calls']==meta['total_diffusion_calls_including_warmup']==0)
check('native_contact_sha',sha(RAW/'native_contacts.json')==meta['native_contacts_sha256'])
check('sensor_sha',sha(RAW/'sensor_diagnostic.npz')==meta['sensor_diagnostic_sha256'])
check('COM_and_collision_geometry_unchanged',meta['native_body_properties']==base_meta['native_body_properties'] and meta['foot_geometry']==base_meta['foot_geometry'] and meta['foot_asset_sha256']==base_meta['foot_asset_sha256'])
foot_com={name:meta['native_body_properties'][0][meta['names'].index(name)]['com'] for name in meta['foot_geometry']}
check('native_COM_finite_per_case_body',len(meta['native_body_properties'])==72 and all(len(p)==24 and all(len(b['com'])==3 and np.isfinite(b['com']).all() and b['mass']>0 for b in p) for p in meta['native_body_properties']))
with np.load(RAW/'trajectories.npz',allow_pickle=False) as a,np.load(OUT/'selected/raw_reference.npz',allow_pickle=False) as s,np.load(BASE/'trajectories.npz',allow_pickle=False) as b:
    check('reference_clock_exact_0_to_119',np.array_equal(a['reference_frame'],np.arange(120)))
    check('selected_goal_float32_exact',np.array_equal(s['original_goal'],np.asarray(base_meta['original_goal'],np.float32)))
    expected=np.zeros((120,72,24,3),np.float32)
    for i in range(72):
        chosen=b['reference'][b['valid'][:,i],i];n=min(len(chosen),120);expected[:n,i]=chosen[:n];expected[n:,i]=chosen[n-1]
    check('selected_frozen_reference_exact_and_terminal_padding',np.array_equal(s['reference'],expected))
    canonical=(s['reference']-s['reference'][:,:,0:1,:])+s['reference'][:,:,0:1,:]
    raw_error=float(np.max(np.abs(a['reference'][a['valid']]-s['reference'][a['valid']])))
    canonical_error=float(np.max(np.abs(a['reference'][a['valid']]-canonical[a['valid']])))
    check('upstream_float32_root_centering_canonical_exact',canonical_error==0 and s['reference'].dtype==np.float32)
terminated=sum(not c['construction_complete'] for c in selection['choices'])
combined=sum(bool(r['joint_pass'] and c['construction_complete']) for r,c in zip(rows,selection['choices']))
check('construction_termination_retained',terminated==2 and combined==0)
ledger=read(OUT/'budget_ledger.json')
check('one_successful_measurement_batch',len(ledger)==1 and ledger[0]['exit_code']==0 and ledger[0]['label']=='val_raw_replay_s11')
check('witness_state',read(OUT/'state.json')['status']=='WITNESS_PASS')
weight_paths={'generator_weights_sha256':'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt','controller_weights_sha256':'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth'}
for key,p in weight_paths.items(): check(key,sha(ROOT/p)==version[key])
upstream=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'rev-parse','HEAD'],text=True).strip()
check('upstream_commit',upstream==version['upstream_commit']==meta['source_commit'])
report={'status':'PASS' if all(checks.values()) else 'FAIL','utc':datetime.now(timezone.utc).isoformat(),'role':'Fresh independent exact-document witness; no retries, source edits, metric changes or Overleaf writes','invocation_count':1,'exit_code':0,'sentinels':['CONTACT_SAMPLING_BATCH_PASS val_raw_replay_s11','FIXED_REFERENCE_COM_DOCUMENT_CHECK_PASS'],'command':'ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python -u /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_fixed_reference_com_v27_20261012.py --check','checks':checks,'file_bindings':file_bindings,'versions_sha256':sha(OUT/'versions.json'),'receipt':receipt,'original_goal_sha256':meta['original_goal_sha256'],'replay_reference_untransformed_max_error_m':raw_error,'replay_canonical_max_error_m':canonical_error,'construction_terminated':terminated,'combined_construction_replay_joint_pass':combined,'native_contacts_sha256':meta['native_contacts_sha256'],'foot_com_local_first_case':foot_com,'ledger':ledger,'scope':'72 fixed raw-reference validation inputs, seed11, one candidate per input. Refined stage and quality efficacy not certified. V24 432 attempts is prerequisite; physical quality floor remains false.','documentation_divergences':[]}
(OUT/'DOCUMENT_WITNESS.json').write_text(json.dumps(report,indent=2),encoding='utf8')
md='# V27 independent document witness\n\nStatus: '+report['status']+'; exact documented `--check` invocation executed once, exit 0; both sentinels observed.\n\nPrerequisites: V26 PID 2763775 already terminal on first inspection; V26 run receipt PASS and preserved V24 outputs verified; V24 six 72-input entries total 432 attempts.\n\nAll '+str(len(checks))+' independent checks '+('passed' if all(checks.values()) else 'did not pass')+'. Source, document, protocol, version, request, selected motion and first-attempt output SHA bindings match. Local and server V27 source/doc hashes also matched before invocation.\n\n72 cases, 24 sources, 3 offsets, seed 11, exactly one original construction candidate each. Selected generated reference equals frozen V24 reference with retained terminal padding. No diffusion calls. Original goal SHA '+meta['original_goal_sha256']+'. Reference clock exactly 0..119. Seven initial state differences exactly zero. Upstream canonical float32 root-centering replay error '+str(canonical_error)+' m (untransformed difference '+str(raw_error)+' m). Native contact SHA verified; measured COMs and collision geometry unchanged.\n\nRaw replay: 71/72 physically completed; 0 joint success. Two construction terminations remain retained as combined failures; combined joint success 0/72. No threshold or source edits, no reruns, no Overleaf writes. No doc/reality divergence found. This certifies only the raw replay interface; it does not certify refined replay, physical quality, efficacy or the full 432-input formal comparison.\n\nExact invocation and all checks/hashes are in DOCUMENT_WITNESS.json.\n'
(OUT/'DOCUMENT_WITNESS.md').write_text(md,encoding='utf8')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'canonical_error_m':canonical_error,'receipt':receipt}))
