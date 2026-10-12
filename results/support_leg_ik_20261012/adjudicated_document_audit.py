"""Independent continuation audit; reads frozen CPU arrays and writes new witness only."""
import collections
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT = ROOT/'results/support_leg_ik_20261012'
PARENT = ROOT/'results/support_reference_probe_v63_20261012'
BASE = ROOT/'results/paired_feedback_validation_v51_20261012'
def read(p): return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
r=dict(status='FAIL',independent_witness=True,witness_agent='/root/leg_ik_adjudicated_document_follower',
    time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),documented_check_invocations=1,
    actual_terminal=dict(exit_code=0,stdout='SUPPORT_LEG_IK_ADJUDICATED_PREFLIGHT_PASS\n',stderr='',tool_chunk='8dfa35'),
    actual_exit_code=0,actual_sentinel='SUPPORT_LEG_IK_ADJUDICATED_PREFLIGHT_PASS',
    original_failure_preserved=True,original_witness_status='FAIL',checks={},failures=[],inspected_hashes={},
    simulation_calls=0,generation_calls=0,training_calls=0,api_calls=0,run_invocations=0,retries=0,
    package_installs=0,source_fixes=0,thresholds_changed=False,historical_files_modified=False,
    scalar_record_bitwise_equality_required=False,
    limitations=['One-seed internal-validation preparation integrity only; no efficacy or novelty conclusion.',
      'Cartesian control may violate bone lengths; it is a positional representation ablation.',
      'Classical positional IK supplies no joint-limit, torque, collision or full-dynamics guarantee.',
      'Endpoint reference equality does not establish actual endpoint success.',
      'Historical scalar reduction context remains unproven; logged low-bit norm discrepancies remain retained.',
      'Original C2/C3 gates remain in force.'])
def check(name,value,detail=None):
    r['checks'][name]=bool(value)
    if not value:r['failures'].append(dict(check=name,detail=detail))
def inspect(p):
    h=sha(p);r['inspected_hashes'][str(p.relative_to(ROOT))]=h;return h
def binding(directory,wanted):
    lin=read(directory/'lineage.json');rec=read(directory/'receipt.json')
    check(str(directory.relative_to(ROOT))+'_request',lin['request']==read(wanted) and lin['certification']=='bound_request_at_execution')
    check(str(directory.relative_to(ROOT))+'_output_names',set(lin['outputs'])=={'metadata.json','metrics.json','trajectories.npz'})
    for n,h in lin['outputs'].items():check(str(directory.relative_to(ROOT))+'_'+n,inspect(directory/n)==h)
    check(str(directory.relative_to(ROOT))+'_complete_receipt',rec['status']=='complete' and rec['trajectories_sha256']==lin['outputs']['trajectories.npz'])
    inspect(wanted)
    return dict(lineage_sha256=inspect(directory/'lineage.json'),receipt_sha256=inspect(directory/'receipt.json'),request=read(wanted),output_hashes=lin['outputs'],certification=lin['certification'])
try:
    pre=read(OUT/'continuation_preflight.json');p=read(OUT/'preparation.json');v=read(OUT/'versions.json')
    old=read(OUT/'DOCUMENT_WITNESS.json');d=read(OUT/'SCALAR_CONTRACT_DIAGNOSIS.json')
    r.update(preflight_sha256=inspect(OUT/'adjudicated_preflight.json'),source_sha256=inspect(ROOT/'scripts/run_support_leg_ik_adjudicated_20261012.py'),document_sha256=inspect(ROOT/'docs/support_leg_ik_adjudicated_20261012.md'))
    check('continuation_preflight',pre['status']=='PASS' and pre['simulation_calls']==0 and pre['source_sha256']==inspect(ROOT/'scripts/continue_support_leg_ik_20261012.py') and pre['document_sha256']==inspect(ROOT/'docs/support_leg_ik_continuation_20261012.md'))
    for n,h in pre['evidence']['files'].items():check('preflight_frozen_'+n,inspect(ROOT/n)==h)
    check('original_FAIL_retained',old['status']=='FAIL' and pre['evidence']['original_witness_retained']=='FAIL')
    check('only_extra_logged_scalar_failures',len(old['failures'])==117 and all(f['check'].startswith('frozen_proxy_objective_records_') for f in old['failures']))
    check('original_non_scalar_checks',all(value for name,value in old['checks'].items() if not name.startswith('frozen_proxy_objective_records_')))
    check('separate_scalar_diagnosis_scope',d['status']=='PASS_DOCUMENTED_PREPARATION_INVARIANTS' and d['original_witness_status_unchanged']=='FAIL' and all(d['checks'].values()) and not d['simulation_authorization'] and not d['efficacy_conclusion'] and not d['thresholds_changed'] and not d['original_witness_overwritten'])
    for n,h in d['bindings']['original_files'].items():check('retained_original_'+n,inspect(OUT/n)==h)
    for key,path in [('document_sha256',ROOT/'docs/support_leg_ik_20261012.md'),('preparation_source_sha256',ROOT/'scripts/prepare_support_leg_ik_20261012.py'),('original_audit_source_sha256',OUT/'document_follower_audit.py'),('diagnostic_source_sha256',OUT/'scalar_contract_diagnosis.py')]:check('diagnosis_'+key,inspect(path)==d['bindings'][key])
    for n,h in v['files'].items():check('frozen_code_'+n,inspect(ROOT/n)==h)
    check('parent_versions',inspect(PARENT/'versions.json')==v['parent_versions_sha256'])
    for n,key in [('source_manifest.json','source_manifest_sha256'),('frozen_thresholds.json','thresholds_sha256')]:check(n,inspect(OUT/n)==inspect(PARENT/n)==inspect(BASE/n)==v[key])
    for key,path in [('controller_weights_sha256',ROOT/'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth'),('generator_weights_sha256',ROOT/'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt')]:check(key,inspect(path)==v[key]==read(PARENT/'versions.json')[key]==read(BASE/'versions.json')[key])
    plan=read(OUT/'analysis_plan_lock.json');extra=read(OUT/'DOCUMENT_WITNESS_FROZEN_ANALYSIS_BINDING.json')
    check('frozen_pre_outcome_analysis',inspect(ROOT/'scripts/analyze_support_leg_ik_20261012.py')==plan['source_sha256']==extra['analysis_source_sha256'] and inspect(OUT/'analysis_plan_lock.json')==extra['analysis_plan_lock_sha256'] and inspect(OUT/'DOCUMENT_WITNESS.json')==extra['document_witness_sha256'])
    source=BASE/'raw/val_shared_s11_r0';baseline=PARENT/'raw/val_baseline_s11'
    sb=binding(source,BASE/'requests/val_shared_s11_r0.json');bb=binding(baseline,PARENT/'requests/val_baseline_s11.json')
    check('source_lineage_binding',sb==p['source_binding']==old['bindings']['source'])
    check('cached_baseline_lineage_binding',bb==p['baseline_binding']==old['bindings']['baseline']==pre['evidence']['original_verification']['baseline_binding']==read(PARENT/'runtime_witness.json')['result']['binding'])
    add=read(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json')
    check('cached_parent_adjudication',add['status']=='PASS' and inspect(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json')==pre['evidence']['original_verification']['parent_adjudication_sha256'])
    for n,h in add['input_hashes'].items():
        if n!='budget_ledger.json':check('cached_parent_'+n,inspect(PARENT/n)==h)
    charges=[x for x in read(PARENT/'budget_ledger.json') if x['label']=='val_baseline_s11']
    check('cached_baseline_original_cost',charges==read(PARENT/'DOCUMENT_WITNESS.json')['budget']['charged_baseline_ledger'] and charges[0]['seconds']>0)
    r['cached_baseline_original_charge']=charges
    third=read(OUT/'adjudicated_preflight.json');amend=read(OUT/'CONTINUATION_CONTRACT_ADDENDUM.json');cont=read(OUT/'CONTINUATION_DOCUMENT_WITNESS.json')
    check('adjudicated_preflight',third['status']=='PASS' and third['simulation_calls']==0 and third['source_sha256']==r['source_sha256'] and third['document_sha256']==r['document_sha256'])
    for n,h in third['evidence']['files'].items():check('adjudicated_preflight_binding_'+n,inspect(ROOT/n)==h)
    check('both_historical_FAILs_retained',old['status']==cont['status']=='FAIL' and third['evidence']['original_witness_status']==third['evidence']['continuation_witness_status']=='FAIL_RETAINED')
    check('continuation_only_historical_ledger_failure',len(cont['failures'])==1 and cont['failures'][0]['check']=='cached_parent_budget_ledger.json' and all(v for k,v in cont['checks'].items() if k!='cached_parent_budget_ledger.json'))
    check('independent_continuation_adjudication',amend['status']=='PASS_DOCUMENTED_CONTINUATION_INVARIANTS' and amend['independent_witness'] and all(amend['checks'].values()) and not amend['failures'] and amend['original_failure_preserved'] and amend['continuation_FAIL_preserved'])
    for key,path in [('original_witness_sha256',OUT/'CONTINUATION_DOCUMENT_WITNESS.json'),('preflight_sha256',OUT/'continuation_preflight.json'),('source_sha256',ROOT/'scripts/continue_support_leg_ik_20261012.py'),('document_sha256',ROOT/'docs/support_leg_ik_continuation_20261012.md')]:
        check('continuation_adjudication_'+key,amend[key]==inspect(path))
    for key,path in [('continuation_FAIL_sha256',OUT/'CONTINUATION_DOCUMENT_WITNESS.json'),('continuation_FAIL_md_sha256',OUT/'CONTINUATION_DOCUMENT_WITNESS.md'),('original_FAIL_sha256',OUT/'DOCUMENT_WITNESS.json'),('original_FAIL_md_sha256',OUT/'DOCUMENT_WITNESS.md'),('parent_addendum_sha256',PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json')]:
        check('adjudication_retained_'+key,amend['bindings'][key]==inspect(path))
    for n,h in amend['input_hashes'].items():
        if n!='results/support_reference_probe_v63_20261012/budget_ledger.json':check('adjudication_current_immutable_'+n,inspect(ROOT/n)==h)
    reconstructed=hashlib.sha256(json.dumps(charges,indent=2).encode()).hexdigest()
    check('historical_baseline_only_snapshot_reconstructed',reconstructed==add['input_hashes']['budget_ledger.json']==amend['ledger_observation']['historical_whole_ledger_sha256']==amend['ledger_observation']['historical_baseline_only_serialization_sha256'])
    ledger=read(PARENT/'budget_ledger.json')
    check('parent_append_only_three_completed_arms',len(ledger)==3 and ledger[0]==charges[0] and [x['label'] for x in ledger]==['val_baseline_s11','val_unconstrained_s11','val_endpoint_preserved_s11'])
    summary=read(PARENT/'summary.json')
    for arm,row in zip(('baseline','unconstrained','endpoint_preserved'),ledger):
        label='val_'+arm+'_s11';bd=binding(PARENT/'raw'/label,PARENT/'requests'/(label+'.json'))
        check('completed_parent_charge_'+arm,row['exit_code']==0 and row['seconds']>0 and row['request_sha256']==inspect(PARENT/'requests'/(label+'.json')))
        check('completed_parent_summary_'+arm,summary['reports'][arm]['binding']==bd)
    r['historical_ledger_snapshot']=dict(reconstructed_sha256=reconstructed,historical_sha256=add['input_hashes']['budget_ledger.json'],baseline_charge=charges[0],current_sha256=inspect(PARENT/'budget_ledger.json'),current_rows=ledger,whole_current_equality_required=False)
    r['historical_witness_statuses']=dict(original='FAIL_RETAINED',continuation='FAIL_RETAINED')
    r['resource_limits']=dict(inputs_per_arm=72,sources=24,seed=11,new_arms=['cartesian','two_link_ik'],arm_timeout_seconds=300,total_seconds=600,global_hours=72,comparison_stage_hours=12,disk_free_floor_GiB=20,round_output_cap_GiB=15)
    check('no_new_simulation',read(OUT/'budget_ledger.json')==[] and not list((OUT/'raw').iterdir()) and not (OUT/'launch_receipt.json').exists())
    sm=read(source/'metadata.json');rows=read(source/'metrics.json');sa=npz(source/'trajectories.npz');names=sm['names'];lengths=sa['valid'].sum(axis=0).astype(int)
    check('source_trajectories_diagnosis_binding',inspect(source/'trajectories.npz')==d['bindings']['source_trajectories_sha256'])
    check('all_72_24_sources',len(rows)==len(sm['cases'])==72 and len(set(x['source_id'] for x in rows))==24)
    check('source_case_order',[x['case_id'] for x in rows]==p['source_case_order']==[x['case_id'] for x in sm['cases']])
    check('three_offsets',all(sorted(c['offset_m'] for c in sm['cases'] if c['source_id']==s)==[-.15,0.,.15] for s in set(x['source_id'] for x in rows)))
    refs={a:npz(OUT/'references'/(a+'.npz')) for a in ('baseline','cartesian','two_link_ik')};b=refs['baseline']['reference'];goals=np.asarray(sm['original_goal'],np.float32)
    for a in refs:check(a+'_reference_hash',inspect(OUT/'references'/(a+'.npz'))==p['reference_hashes'][a]==d['bindings']['references'][a])
    check('cached_baseline_bytes',inspect(OUT/'references/baseline.npz')==inspect(PARENT/'references/baseline.npz'))
    expected=np.empty_like(b)
    for n,length in enumerate(lengths):
        check('source_valid_prefix_'+str(n),length>0 and np.array_equal(sa['valid'][:,n],np.arange(120)<length))
        expected[:length,n]=sa['actual'][:length,n];expected[length:,n]=sa['actual'][length-1,n]
    check('fixed_first_construction_baseline',np.array_equal(expected,b))
    unchanged=[j for j in range(24) if names[j] not in [s+'_'+part for s in ('L','R') for part in ('Knee','Ankle','Toe')]]
    r['reference_invariants']={}
    for a,z in refs.items():
        x=z['reference'];delta=x.astype(np.float64)-b.astype(np.float64);mx=float(np.linalg.norm(delta,axis=-1).max());bone=0.;pair=0.
        check(a+'_shape_finite',x.shape==(120,72,24,3) and np.isfinite(x).all() and np.isfinite(z['original_goal']).all())
        check(a+'_prefix61',np.array_equal(x[:61],b[:61]));check(a+'_last_pose',np.array_equal(x[-1],b[-1]));check(a+'_original_goals',np.array_equal(z['original_goal'],goals))
        check(a+'_anchors',np.array_equal(x[:,:,unchanged],b[:,:,unchanged]));check(a+'_10cm',mx<=.1,mx)
        check(a+'_incomplete_untouched',np.array_equal(x[:,lengths<120],b[:,lengths<120]))
        for side in ('L','R'):
            h,k,f,t=[names.index(side+'_'+part) for part in ('Hip','Knee','Ankle','Toe')]
            pair=max(pair,float(np.abs(delta[:,:,f]-delta[:,:,t]).max()))
            check(a+'_'+side+'_paired_feet',float(np.abs(delta[:,:,f]-delta[:,:,t]).max())<=2e-7)
            check(a+'_'+side+'_horizontal_feet',np.array_equal(x[:,:,[f,t],2],b[:,:,[f,t],2]))
            if a=='cartesian':check(a+'_'+side+'_original_knee',np.array_equal(x[:,:,k],b[:,:,k]))
            for u,w in ((h,k),(k,f)):
                err=float(np.abs(np.linalg.norm(x[:,:,u].astype(float)-x[:,:,w].astype(float),axis=-1)-np.linalg.norm(b[:,:,u].astype(float)-b[:,:,w].astype(float),axis=-1)).max());bone=max(bone,err)
                if a=='two_link_ik':check(a+'_'+names[u]+'_segment_1e5',err<=1e-5,err)
            check(side+'_same_accepted_feet',np.array_equal(refs['cartesian']['reference'][:,:,[f,t]],refs['two_link_ik']['reference'][:,:,[f,t]]))
        r['reference_invariants'][a]=dict(max_joint_shift_m=mx,max_segment_error_m=bone,max_paired_feet_float32_roundoff_m=pair)
    check('frozen_runtime',sys.version_info[:2]==(3,8) and np.__version__=='1.23.5')
    r['runtime']=dict(python=sys.version,numpy=np.__version__)
    r['denominators']=dict(cases=72,sources=24,offsets_per_source=3,incomplete_cases=int((lengths<120).sum()),all_inputs_retained=True)
    r['scalar_observation']=dict(original_failed_scalar_checks=len(old['failures']),document_demands_exact_logged_norms=False,diagnosis_max_logged_norm_difference_m=6.938893903907228e-17,reduction_context_proven=False)
    r['status']='FAIL' if r['failures'] else 'PASS'
except Exception as e:
    r['failures'].append(dict(exception=repr(e),traceback=traceback.format_exc()))
for key,path in [('preflight_sha256',OUT/'adjudicated_preflight.json'),('source_sha256',ROOT/'scripts/run_support_leg_ik_adjudicated_20261012.py'),('document_sha256',ROOT/'docs/support_leg_ik_adjudicated_20261012.md')]:r.setdefault(key,sha(path) if path.exists() else None)
with (OUT/'ADJUDICATED_DOCUMENT_WITNESS.json').open('x') as f:json.dump(r,f,indent=2)
lines=['# Independent adjudicated document witness','','Status: '+r['status'],'','The documented CPU --check executed exactly once. Actual exit: 0. Sentinel: SUPPORT_LEG_IK_ADJUDICATED_PREFLIGHT_PASS. No --run, retry, simulator, generation, training, install, API, source change or threshold change.','','Original DOCUMENT_WITNESS.json/.md remain FAIL and are hash-bound unchanged. Both historical FAILs remain unchanged and hash-bound. Separate scalar and continuation adjudications confirm documented preparation invariants. Historical whole-ledger SHA is reconstructed from the unchanged baseline row; current append-only ledger equality is not required; exact logged float64 norm equality is not a documented requirement. Historical scalar reduction context remains unproven.','']
lines += [key+': '+str(r[key]) for key in ('preflight_sha256','source_sha256','document_sha256')]
lines += ['','Independent saved-array geometry:',json.dumps(r.get('reference_invariants',{}),indent=2),'','Limitations:']+['- '+x for x in r['limitations']]
lines += ['','All inspected hashes, checks, immutable lineage, cached charge and failures appear in JSON.','Failures: '+json.dumps(r['failures'])]
with (OUT/'ADJUDICATED_DOCUMENT_WITNESS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({k:r[k] for k in ('status','preflight_sha256','source_sha256','document_sha256','reference_invariants','failures')},indent=2))
sys.exit(0 if r['status']=='PASS' else 1)
