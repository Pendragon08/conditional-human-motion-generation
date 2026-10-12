"""One-shot independent CPU document follower; no simulator or model invocation."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT=ROOT/'results/support_leg_ik_20261012'
PARENT=ROOT/'results/support_reference_probe_v63_20261012'
BASE=ROOT/'results/paired_feedback_validation_v51_20261012'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
report=dict(status='FAIL',independent_witness=True,witness_agent='/root/support_leg_ik_document_follower',
    time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),documented_check_invocations=1,
    actual_terminal=dict(exit_code=0,stdout='SUPPORT_LEG_IK_DOCUMENT_PREFLIGHT_PASS\n',stderr='',tool_chunk='634fea'),
    simulation_calls=0,generation_calls=0,training_calls=0,api_calls=0,run_invocations=0,retries=0,
    fixes=0,package_installs=0,overleaf_results_sync=False,
    scope='CPU document/reference/lineage preflight only; not efficacy, novelty, innovation, physical success or downstream benefit.',
    checks={},failures=[])
report['invocation']='ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/scripts/run_support_leg_ik_20261012.py --check'
def check(name,value,detail=None):
    report['checks'][name]=bool(value)
    if not value:report['failures'].append(dict(check=name,detail=detail))
def binding(directory,request_path):
    lin=read(directory/'lineage.json');rec=read(directory/'receipt.json');wanted=read(request_path)
    check(str(directory.relative_to(ROOT))+'_execution_binding',lin['certification']=='bound_request_at_execution' and lin['request']==wanted)
    check(str(directory.relative_to(ROOT))+'_complete_bound_output_names',set(lin['outputs'])=={'metadata.json','metrics.json','trajectories.npz'})
    for name,digest in lin['outputs'].items():check(str(directory.relative_to(ROOT))+'_'+name+'_sha256',sha(directory/name)==digest)
    check(str(directory.relative_to(ROOT))+'_receipt',rec['status']=='complete' and rec['trajectories_sha256']==lin['outputs']['trajectories.npz'])
    return dict(lineage_sha256=sha(directory/'lineage.json'),receipt_sha256=sha(directory/'receipt.json'),request=wanted,output_hashes=lin['outputs'],certification=lin['certification'])
try:
    v=read(OUT/'versions.json');p=read(OUT/'preparation.json');pre=read(OUT/'preflight.json')
    report.update(preflight_sha256=sha(OUT/'preflight.json'),source_sha256=sha(ROOT/'scripts/run_support_leg_ik_20261012.py'),document_sha256=sha(ROOT/'docs/support_leg_ik_20261012.md'),own_audit_script_sha256=sha(__file__))
    check('preflight_binding',pre['status']=='PASS' and pre['simulation_calls']==pre['generation_calls']==0 and pre['source_sha256']==report['source_sha256'] and pre['document_sha256']==report['document_sha256'])
    report['source_code_hashes']={name:sha(ROOT/name) for name in v['files']}
    for name,h in v['files'].items():check('frozen_code_'+name,report['source_code_hashes'][name]==h)
    check('parent_versions',sha(PARENT/'versions.json')==v['parent_versions_sha256'])
    check('manifest_binding',sha(OUT/'source_manifest.json')==sha(PARENT/'source_manifest.json')==sha(BASE/'source_manifest.json')==v['source_manifest_sha256'])
    check('threshold_binding',sha(OUT/'frozen_thresholds.json')==sha(PARENT/'frozen_thresholds.json')==sha(BASE/'frozen_thresholds.json')==v['thresholds_sha256'])
    weights={'controller_weights_sha256':ROOT/'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth','generator_weights_sha256':ROOT/'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt'}
    report['model_hashes']={k:sha(path) for k,path in weights.items()}
    for k,h in report['model_hashes'].items():check('frozen_'+k,h==v[k]==read(PARENT/'versions.json')[k]==read(BASE/'versions.json')[k])
    spec=read(ROOT/'configs/env-closd-baseline-20261010.json')
    spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'diff','--name-only','HEAD','--','closd'],text=True).strip()
    check('pinned_runtime',sys.version_info[:2]==(3,8) and np.__version__=='1.23.5' and spec_hash=='81bdf764e312bcc92e4969e5eb9d2d6d6043f14c037e5a588dabbd564871f780' and commit==spec['source_commit'] and not dirty)
    report['runtime']=dict(python=sys.version,numpy=np.__version__,spec_sha256=spec_hash,closd_commit=commit,closd_diff_names=dirty,packages_installed=0)
    source=BASE/'raw/val_shared_s11_r0';baseline=PARENT/'raw/val_baseline_s11'
    source_binding=binding(source,BASE/'requests/val_shared_s11_r0.json');baseline_binding=binding(baseline,PARENT/'requests/val_baseline_s11.json')
    check('prepared_source_binding',source_binding==p['source_binding'])
    check('prepared_baseline_binding',baseline_binding==p['baseline_binding']==pre['evidence']['baseline_binding']==read(PARENT/'runtime_witness.json')['result']['binding'])
    report['bindings']=dict(source=source_binding,baseline=baseline_binding,baseline_request_sha256=sha(PARENT/'requests/val_baseline_s11.json'),versions_sha256=sha(OUT/'versions.json'),preparation_sha256=sha(OUT/'preparation.json'))
    add=read(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json');old=read(PARENT/'DOCUMENT_WITNESS.json')
    check('original_fail_retained',old['status']=='FAIL' and sha(PARENT/'DOCUMENT_WITNESS.json')==add['original_FAIL_witness_sha256'] and sha(PARENT/'DOCUMENT_WITNESS_INITIAL_FAIL.json')==add['original_INITIAL_FAIL_witness_sha256'])
    check('separate_parent_adjudication',add['status']=='PASS' and add['versions_sha256']==sha(PARENT/'versions.json') and add['runtime_witness_sha256']==sha(PARENT/'runtime_witness.json') and pre['evidence']['parent_adjudication_sha256']==sha(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json'))
    frozen_parent_inputs=['DOCUMENT_WITNESS.json','DOCUMENT_WITNESS.md','DOCUMENT_WITNESS_INITIAL_FAIL.json','DOCUMENT_WITNESS_INITIAL_FAIL.md','versions.json','runtime_witness.json','raw/val_baseline_s11/metadata.json','raw/val_baseline_s11/metrics.json','raw/val_baseline_s11/trajectories.npz','raw/val_baseline_s11/receipt.json','raw/val_baseline_s11/lineage.json','requests/val_baseline_s11.json','references/baseline.npz']
    for name in frozen_parent_inputs:check('retained_parent_'+name,sha(PARENT/name)==add['input_hashes'][name])
    ledger=read(PARENT/'budget_ledger.json');charged=[r for r in ledger if r['label']=='val_baseline_s11'];initial_charge=old['budget']['charged_baseline_ledger']
    check('baseline_original_charge_retained',charged==initial_charge and charged[0]['seconds']>0)
    check('empty_new_simulator_ledger',read(OUT/'budget_ledger.json')==[] and not any((OUT/'raw').iterdir()) and not (OUT/'launch_receipt.json').exists())
    report['parent']=dict(adjudication_sha256=sha(PARENT/'NUMERICAL_CONTRACT_ADDENDUM.json'),original_fail_sha256=sha(PARENT/'DOCUMENT_WITNESS.json'),initial_fail_sha256=sha(PARENT/'DOCUMENT_WITNESS_INITIAL_FAIL.json'),original_failed_checks=old['all_failed_checks'],baseline_original_charged_rows=charged,numerical_adjudication_ledger_sha256=sha(PARENT/'numerical_adjudication_budget_ledger.json'))
    sm=read(source/'metadata.json');sr=read(source/'metrics.json');sa=npz(source/'trajectories.npz');names=sm['names'];cases=sm['cases'];lengths=sa['valid'].sum(axis=0).astype(int)
    check('all_72_24_sources',len(sr)==len(cases)==72 and len(set(r['source_id'] for r in sr))==24)
    order=[r['case_id'] for r in sr]
    check('case_order',order==p['source_case_order']==[c['case_id'] for c in cases])
    counts=collections.Counter(c['source_id'] for c in cases)
    check('three_offsets_each_source',all(n==3 for n in counts.values()) and all(sorted(c['offset_m'] for c in cases if c['source_id']==s)==[-.15,0.,.15] for s in counts))
    manifest=read(OUT/'source_manifest.json');cfg=read(ROOT/'configs/physics_guided_20261011.json');wanted=[]
    for row in manifest['rows']:
        if row['split']=='val':
            for offset in cfg['offsets_m']:wanted.append(dict(row,offset_m=offset,goal_joint_body='Pelvis' if row['goal_joint']=='pelvis' else 'R_Wrist',case_id=row['clip_id']+':'+str(offset)))
    check('source_case_text_task_binding',wanted==cases and all(c['goal_joint_body'] in ('Pelvis','L_Wrist','R_Wrist') for c in cases))
    refs={a:npz(OUT/'references'/(a+'.npz')) for a in ('baseline','cartesian','two_link_ik')};ref=refs['baseline']['reference'];goal=np.asarray(sm['original_goal'],np.float32)
    report['reference_hashes']={a:sha(OUT/'references'/(a+'.npz')) for a in refs}
    check('reference_hash_bindings',report['reference_hashes']==p['reference_hashes'])
    check('baseline_byte_identical',report['reference_hashes']['baseline']==sha(PARENT/'references/baseline.npz'))
    expected=np.empty((120,72,24,3),np.float32)
    for n,length in enumerate(lengths):
        check('valid_prefix_'+str(n),length>0 and np.array_equal(sa['valid'][:,n],np.arange(120)<length))
        expected[:length,n]=sa['actual'][:length,n];expected[length:,n]=sa['actual'][length-1,n]
    check('baseline_fixed_first_construction',np.array_equal(ref,expected))
    check('goals_source_hash',arrsha(goal)==p['source_original_goal_sha256']==sm['original_goal_sha256'])
    changed=[names.index(s+'_'+j) for s in ('L','R') for j in ('Knee','Ankle','Toe')];unchanged=[i for i in range(24) if i not in changed]
    report['reference_invariants']={};report['per_case_reference_array_sha256']={}
    for arm,z in refs.items():
        x=z['reference'];d=x.astype(np.float64)-ref.astype(np.float64);mx=float(np.linalg.norm(d,axis=-1).max());maxbone=0.;maxpair=0.
        check(arm+'_shape_finite',x.shape==(120,72,24,3) and np.isfinite(x).all() and np.isfinite(z['original_goal']).all())
        check(arm+'_prefix61',np.array_equal(x[:61],ref[:61]));check(arm+'_endpose',np.array_equal(x[-1],ref[-1]));check(arm+'_all_targets',np.array_equal(z['original_goal'],goal))
        check(arm+'_task_upper_root_heading_anchors',np.array_equal(x[:,:,unchanged],ref[:,:,unchanged]))
        check(arm+'_all_joint_shift_10cm',mx<=.1,dict(observed_m=mx,criterion_m=.1))
        check(arm+'_incomplete_sources_untouched',np.array_equal(x[:,lengths<120],ref[:,lengths<120]))
        for side in ('L','R'):
            h,k,a,t=[names.index(side+'_'+j) for j in ('Hip','Knee','Ankle','Toe')]
            pair=float(np.abs(d[:,:,a]-d[:,:,t]).max());maxpair=max(maxpair,pair)
            check(arm+'_'+side+'_paired_ankle_toe',pair<=2e-7,dict(observed_float32_position_roundoff_m=pair))
            check(arm+'_'+side+'_feet_horizontal',np.array_equal(x[:,:,[a,t],2],ref[:,:,[a,t],2]))
            for u,w in ((h,k),(k,a)):
                b0=np.linalg.norm(ref[:,:,u].astype(np.float64)-ref[:,:,w].astype(np.float64),axis=-1)
                b1=np.linalg.norm(x[:,:,u].astype(np.float64)-x[:,:,w].astype(np.float64),axis=-1)
                err=float(np.abs(b1-b0).max());maxbone=max(maxbone,err)
                if arm=='two_link_ik':check(arm+'_'+side+'_'+names[u]+'_per_frame_length_1e5',err<=1e-5,dict(observed_m=err))
            if arm=='cartesian':check(arm+'_'+side+'_knee_original',np.array_equal(x[:,:,k],ref[:,:,k]))
        report['reference_invariants'][arm]=dict(max_joint_change_m=mx,max_paired_ankle_toe_roundoff_m=maxpair,max_per_frame_thigh_calf_error_m=maxbone,first61_exact=True,endpose_exact=True,all_goals_exact=True,incomplete_cases_untouched=True)
        report['per_case_reference_array_sha256'][arm]={case:arrsha(x[:,n]) for n,case in enumerate(order)}
    for side in ('L','R'):
        feet=[names.index(side+'_'+j) for j in ('Ankle','Toe')]
        check('arms_same_accepted_'+side+'_feet',np.array_equal(refs['cartesian']['reference'][:,:,feet],refs['two_link_ik']['reference'][:,:,feet]))
    # Reproduce source-driven requested translations and accepted dyadic solutions
    # with frozen functions; the full coordinate and segment audit above is independent.
    sys.path.insert(0,str(ROOT/'scripts'))
    from support_reference_repair_core_v63_20261012 import translation
    from support_leg_ik_core_20261012 import knee_solution
    from physics_guided_metric_v23_20261012 import rotate_xyzw
    thresholds=read(OUT/'frozen_thresholds.json')['thresholds'];record_map={(r['case_id'],r['side']):r for r in p['records']}
    check('records_all_144_legs',len(p['records'])==len(record_map)==144)
    reconstructed=ref.copy();cart_expected=ref.copy()
    for n,row in enumerate(sr):
        length=lengths[n];start=sm['context_steps']
        for side in ('L','R'):
            h,k,a,t=[names.index(side+'_'+j) for j in ('Hip','Knee','Ankle','Toe')];velocity=np.zeros((120,2,2));support=np.zeros((120,2),bool)
            for j,body in enumerate((a,t)):
                geom=sm['foot_geometry'][names[body]];idx=np.arange(length);q=sa['quaternion'][idx,n,body];position=sa['actual'][idx,n,body]
                points=position[:,None]+rotate_xyzw(q[:,None],np.asarray(geom['points'])[None]);points[:,:,2]-=np.asarray(geom['radii'])[None]
                low=points[np.arange(length),np.argmin(points[:,:,2],axis=1)];com=rotate_xyzw(q,np.asarray(sm['native_body_properties'][n][body]['com']))
                velocity[:length,j]=(sa['linvel'][idx,n,body]+np.cross(sa['angvel'][idx,n,body],low-position-com))[:,:2]
                support[:length,j]=(sa['contact_force'][idx,n,body,2]>thresholds['contact_force_n'])&(low[:,2]<thresholds['surface_near_ground_m'])
            requested=translation(velocity,support,start,1/30,True) if length==120 else np.zeros((120,2));saved=record_map[(row['case_id'],side)]['frames'];correct=len(saved)==58
            for f in range(start+1,119):
                knee,delta,alpha=knee_solution(ref[f,n,h],ref[f,n,k],ref[f,n,a],np.r_[requested[f],0.])
                for body in (a,t):cart_expected[f,n,body]=ref[f,n,body]+delta.astype(np.float32);reconstructed[f,n,body]=cart_expected[f,n,body]
                reconstructed[f,n,k]=knee.astype(np.float32);r=saved[f-start-1]
                correct=correct and r['frame']==f and r['scale']==alpha and r['accepted_shift_m']==float(np.linalg.norm(delta)) and (alpha in (0.,1.) or alpha in {2.**(-e) for e in range(1,41)})
            check('frozen_proxy_objective_records_'+row['case_id']+'_'+side,correct)
    check('cartesian_matches_frozen_source_proxy_objective',np.array_equal(cart_expected,refs['cartesian']['reference']))
    check('ik_matches_frozen_source_proxy_objective_and_dyadic_backoff',np.array_equal(reconstructed,refs['two_link_ik']['reference']))
    report['denominators']=dict(reference_cases_per_arm=72,source_clusters=24,offsets_per_source=3,legs=144,full_length_source_cases=int((lengths==120).sum()),incomplete_source_cases=int((lengths<120).sum()),all_cases_retained=True)
    report['status']='FAIL' if report['failures'] else 'PASS'
except Exception as e:
    report['failures'].append(dict(exception=repr(e),traceback=traceback.format_exc()))
    report['status']='FAIL'
for field,path in [('preflight_sha256',OUT/'preflight.json'),('source_sha256',ROOT/'scripts/run_support_leg_ik_20261012.py'),('document_sha256',ROOT/'docs/support_leg_ik_20261012.md')]:
    if field not in report:report[field]=sha(path) if path.exists() else None
with (OUT/'DOCUMENT_WITNESS.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
lines=['# Independent support-leg IK document witness','', 'Status: '+report['status'],'','Exact documented CPU --check executed once. Terminal exit: 0. Stdout: SUPPORT_LEG_IK_DOCUMENT_PREFLIGHT_PASS. No retry, --run, simulator, source fix, install or Overleaf write.','', 'This binds preparation integrity only and does not establish efficacy, physical success, novelty or downstream gain.','']
for field in ('preflight_sha256','source_sha256','document_sha256'):lines.append(field+': '+str(report[field]))
lines+=['','All reference bundle SHA-256:']
for arm,h in report.get('reference_hashes',{}).items():lines.append('- '+arm+': '+h)
lines+=['','All 72 per-case reference array hashes, all checks, source/code/model hashes, retained parent FAIL and separate adjudication bindings appear in JSON.','',json.dumps(report.get('reference_invariants',{}),indent=2),'','Failures: '+json.dumps(report['failures'],ensure_ascii=False)]
with (OUT/'DOCUMENT_WITNESS.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(status=report['status'],failed_checks=report['failures'],reference_hashes=report.get('reference_hashes'),reference_invariants=report.get('reference_invariants'),preflight_sha256=report['preflight_sha256'],source_sha256=report['source_sha256'],document_sha256=report['document_sha256']),indent=2))
sys.exit(0 if report['status']=='PASS' else 1)
