"""Independent read-only output audit. Does not launch the experiment or mutate receipts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT=ROOT/'results/condition_feedback_formal_v28_20261012'
NOW=OUT/'raw/val_planned_s11_r0'
OLD=ROOT/'results/contact_com_validation_v24_20261012/raw/val_planned_s11'
sys.path.insert(0,str(ROOT/'scripts'))
from physics_guided_metric_v23_20261012 import metrics
from metric_numeric_equivalence import compare_metrics

def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def object_hash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

e=dict(scope='One documented --check and independent read-only output audit; no efficacy inference',audit_time_unix=time.time(),checks={},divergences=[])
def check(name,actual,expected):
    ok=actual==expected
    e['checks'][name]=dict(pass_check=ok,actual=actual,expected=expected)
    if not ok: e['divergences'].append(name)

v=read(OUT/'versions.json'); req=read(OUT/'requests/val_planned_s11_r0.json'); lineage=read(NOW/'lineage.json')
m=read(NOW/'metadata.json'); oldm=read(OLD/'metadata.json'); saved=read(NOW/'metrics.json'); olds=read(OLD/'metrics.json')
cfg=read(ROOT/'configs/physics_guided_20261011.json'); manifest=read(OUT/'source_manifest.json')
e['bound_files']={k:dict(expected=h,actual=sha(ROOT/k)) for k,h in v['files'].items()}
check('all_source_document_config_hashes_match',all(x['expected']==x['actual'] for x in e['bound_files'].values()),True)
check('document_is_in_version_binding','docs/condition_feedback_formal_v28_20261012.md' in v['files'],True)
check('manifest_hash',sha(OUT/'source_manifest.json'),v['manifest_sha256'])
check('manifest_identical_original',sha(OUT/'source_manifest.json'),sha(ROOT/'results/physics_guided_20261011/source_manifest.json'))
check('thresholds_identical_v24',sha(OUT/'frozen_thresholds.json'),sha(ROOT/'results/contact_com_validation_v24_20261012/frozen_thresholds.json'))
check('thresholds_equal_config',read(OUT/'frozen_thresholds.json')['thresholds'],cfg['base_thresholds'])
check('protocol_config_hash',sha(ROOT/'configs/physics_guided_20261011.json'),v['protocol_config_sha256'])
check('prior_budget_hash',sha(OUT/'prior_budget_snapshot.json'),v['prior_budget_snapshot_sha256'])
check('fixed_reference_hash',sha(ROOT/'results/fixed_reference_com_v27_20261012/replay_summary.json'),v['fixed_reference_calibration_sha256'])
check('execution_request',lineage['request'],req)
check('execution_binding_type',lineage['certification'],'bound_request_at_execution')
for key,path in [('config',ROOT/'configs/physics_guided_20261011.json'),('manifest',OUT/'source_manifest.json'),('thresholds',OUT/'frozen_thresholds.json'),('versions',OUT/'versions.json')]: check('request_'+key,sha(path),req[key])
for name,h in lineage['outputs'].items(): check('output_binding_'+name,sha(NOW/name),h)
check('receipt_trajectory_hash',sha(NOW/'trajectories.npz'),read(NOW/'receipt.json')['trajectories_sha256'])
source=ROOT/'third_party/CLoSD'
check('upstream_commit',subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),v['upstream_commit'])
check('upstream_closd_diff',subprocess.check_output(['git','-C',str(source),'diff','--name-only','HEAD','--','closd'],text=True),'')
weights=dict(generator_weights_sha256=source/'closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt',controller_weights_sha256=source/'output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth',text_motion_evaluator_weights_sha256=Path(manifest['snapshot'])/'evaluators/t2m/text_mot_match/model/finest.tar')
for key,p in weights.items(): check(key,sha(p),v[key])
e['external_inputs']={}
for section in ('external_motion_inputs','external_semantic_inputs'):
    e['external_inputs'][section]={p:dict(expected=h,actual=sha(Path(p))) for p,h in v[section].items()}
    check(section+'_hashes',all(x['expected']==x['actual'] for x in e['external_inputs'][section].values()),True)
e['exact_array_fields']={}
for name in ('trajectories.npz','sensor_diagnostic.npz'):
    with np.load(NOW/name,allow_pickle=False) as a,np.load(OLD/name,allow_pickle=False) as b:
        check(name+'_ordered_keys',a.files,b.files)
        for key in a.files:
            x=a[key]; y=b[key]
            same=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
            e['exact_array_fields'][name+':'+key]=dict(bit_identical=bool(same),dtype=str(x.dtype),shape=list(x.shape))
        if name=='trajectories.npz': arrays={k:a[k] for k in a.files}
check('all_array_fields_bit_identical',all(x['bit_identical'] for x in e['exact_array_fields'].values()),True)
check('metrics_equal_v24',saved,olds)
check('metrics_file_hash_identical_v24',sha(NOW/'metrics.json'),sha(OLD/'metrics.json'))
recomputed=[metrics(arrays,i,m,read(OUT/'frozen_thresholds.json')['thresholds']) for i in range(len(saved))]
e['recomputed_numeric_certificate']=compare_metrics(recomputed,saved)
check('metric_verifier_classifications_identical',e['recomputed_numeric_certificate']['classifications_identical'],True)
check('saved_numeric_verifier_status',read(NOW/'metric_numerics.json')['status'],'PASS_ROUNDOFF_ONLY')
check('original_goal_hash_unchanged',m['original_goal_sha256'],oldm['original_goal_sha256'])
check('original_goal_hash_recomputed',hashlib.sha256(np.asarray(m['original_goal'],np.float32).tobytes()).hexdigest(),m['original_goal_sha256'])
for key in ('original_goal','original_heading','cases','episode_steps','dt','context_steps','names','native_body_properties'): check(key+'_unchanged',m[key],oldm[key])
check('dense_recording_disabled',m['dense_manifold_recording'],False)
native=read(NOW/'native_contacts.json')
check('native_contacts_binding',sha(NOW/'native_contacts.json'),m['native_contacts_sha256'])
check('disabled_dense_records_count',len(native['records']),0)
check('native_com_shape',[len(m['native_body_properties']),sorted(set(len(r) for r in m['native_body_properties']))],[72,[24]])
check('native_com_all_finite',bool(np.isfinite(np.asarray([[b['com'] for b in row] for row in m['native_body_properties']])).all()),True)
check('reference_clock',arrays['reference_frame'].tolist(),list(range(cfg['episode_steps_30hz'])))
expected_cases=[]
for row in manifest['rows']:
    if row['split']=='val':
        for offset in cfg['offsets_m']: expected_cases.append(dict(row,offset_m=offset,goal_joint_body='Pelvis' if row['goal_joint']=='pelvis' else 'R_Wrist',case_id=row['clip_id']+':'+str(offset)))
check('source_case_expansion_original',object_hash(m['cases']),object_hash(expected_cases))
check('validation_cases_retained',len(m['cases']),72)
check('test_preselected_sources',len([r for r in manifest['rows'] if r['split']=='test']),48)
check('test_inputs_per_arm',v['test_inputs_per_arm'],48*len(cfg['offsets_m'])*len(cfg['seeds']))
ledger=read(OUT/'budget_ledger.json'); e['ledger']=ledger
check('one_measurement_entry',len(ledger),1)
check('measurement_label',ledger[0]['label'],'val_planned_s11_r0')
check('measurement_stage',ledger[0]['stage'],'measurement')
check('measurement_exit',ledger[0]['exit_code'],0)
check('measurement_stop_reason',ledger[0]['stop_reason'],None)
check('runtime_witness_status',read(OUT/'runtime_witness.json')['status'],'PASS')
check('queue_status',read(OUT/'queue_state.json')['status'],'WITNESS_PASS')
e['result_counts']=dict(cases=len(saved),completed=sum(r['completed'] for r in saved),terminated=sum(r['terminated'] for r in saved),lengths=sorted(set(r['length'] for r in saved)))
e['artifact_sha256']={str(p.relative_to(OUT)):sha(p) for p in [OUT/'versions.json',OUT/'source_manifest.json',OUT/'frozen_thresholds.json',OUT/'budget_ledger.json',OUT/'runtime_witness.json',NOW/'receipt.json',NOW/'lineage.json',NOW/'metadata.json',NOW/'metrics.json',NOW/'trajectories.npz',NOW/'sensor_diagnostic.npz',NOW/'native_contacts.json']}
# Exclude large repeated goal/case/metric payloads from this compact evidence file.
for name in list(e['checks']):
    c=e['checks'][name]
    if len(json.dumps(c))>2000:
        c['actual_object_sha256']=object_hash(c.pop('actual'));c['expected_object_sha256']=object_hash(c.pop('expected'))
e['notes']=['versions.json inherits evaluation_inputs_per_arm=72 from the validation version; the V28-specific test_inputs_per_arm=432 matches the original 48x3x3 protocol.', 'Dense records are empty because recording is disabled, not because there were no physical contacts.', 'The audit verifies frozen calibrated proxy computation and output equality; it does not establish C1 efficacy, semantic fidelity, hardware validity or innovation.', 'No --run, source edits, threshold changes or retries performed by this witness.']
e['status']='PASS' if not e['divergences'] else 'DIVERGENCE'
dest=OUT/'DOCUMENT_WITNESS_AUDIT.json'
with dest.open('x') as f: json.dump(e,f,indent=2,ensure_ascii=False)
print(json.dumps(dict(status=e['status'],checks=len(e['checks']),array_fields=len(e['exact_array_fields']),divergences=e['divergences'],artifact=str(dest),result_counts=e['result_counts'],numeric_certificate={k:v for k,v in e['recomputed_numeric_certificate'].items() if k!='differences'}),ensure_ascii=False))
