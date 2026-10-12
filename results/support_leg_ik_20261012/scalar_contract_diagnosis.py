"""Separate read-only CPU diagnosis; original witness and source never written."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
OUT=ROOT/'results/support_leg_ik_20261012';BASE=ROOT/'results/paired_feedback_validation_v51_20261012'
sys.path.insert(0,str(ROOT/'scripts'))
from support_reference_repair_core_v63_20261012 import translation
from support_leg_ik_core_20261012 import knee_solution
from physics_guided_metric_v23_20261012 import rotate_xyzw
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def check(name,v):checks[name]=bool(v)
def get_loop(path,iterator):
    tree=ast.parse(path.read_text(encoding='utf8'))
    return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='enumerate' and isinstance(n.iter.args[0],ast.Name) and n.iter.args[0].id==iterator)
def execute(loop,scope,label):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])),label,'exec'),scope)
def compare(original,computed):
    mismatches=[];legs=set();maxabs={};counts={};total=0
    for saved,new in zip(original,computed):
        for a,b in zip(saved['frames'],new['frames']):
            total+=1
            for field in ('frame','scale','accepted_shift_m','bone_error_m'):
                if field not in b:continue
                x,y=a[field],b[field];err=abs(float(x)-float(y));maxabs[field]=max(maxabs.get(field,0.),err)
                if x!=y:
                    legs.add((saved['case_id'],saved['side']));counts[field]=counts.get(field,0)+1
                    mismatches.append(dict(case_id=saved['case_id'],side=saved['side'],frame=a['frame'],field=field,saved=x,recomputed=y,absolute_error=err,saved_hex=float(x).hex(),recomputed_hex=float(y).hex()))
    return dict(compared_leg_frames=total,mismatched_legs=len(legs),mismatched_fields=len(mismatches),mismatch_counts=counts,max_absolute_difference=maxabs,all_mismatches=mismatches)
before={n:sha(OUT/n) for n in ('DOCUMENT_WITNESS.json','DOCUMENT_WITNESS.md','preflight.json','preparation.json','versions.json')}
checks={};w=read(OUT/'DOCUMENT_WITNESS.json');p=read(OUT/'preparation.json');sm=read(BASE/'raw/val_shared_s11_r0/metadata.json');sr=read(BASE/'raw/val_shared_s11_r0/metrics.json');sa=load(BASE/'raw/val_shared_s11_r0/trajectories.npz');refs={a:load(OUT/'references'/(a+'.npz')) for a in ('baseline','cartesian','two_link_ik')};ref=refs['baseline']['reference'];names=sm['names'];thresholds=read(OUT/'frozen_thresholds.json')['thresholds']
prepare_path=ROOT/'scripts/prepare_support_leg_ik_20261012.py';audit_path=OUT/'document_follower_audit.py';doc_path=ROOT/'docs/support_leg_ik_20261012.md'
prepare_scope=dict(np=np,metadata=sm,rows=sr,arrays=sa,refs=ref.copy(),thresholds=thresholds,cartesian=ref.copy(),ik=ref.copy(),records=[],rotate_xyzw=rotate_xyzw,translation=translation,knee_solution=knee_solution)
prepare_loop=get_loop(prepare_path,'rows');execute(prepare_loop,prepare_scope,str(prepare_path)+'::read_only_loop_only')
prepared_comparison=compare(p['records'],prepare_scope['records'])
# Extract the original follower's numerical loop only. Instrument in memory to
# retain exact observed scalar values; no witness/code file is changed.
audit_loop=get_loop(audit_path,'sr')
inner=next(n for n in ast.walk(audit_loop) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='f')
inner.body.append(ast.parse("observed[(row['case_id'],side)].append(dict(frame=f,scale=alpha,accepted_shift_m=float(np.linalg.norm(delta))))").body[0])
side_loop=next(n for n in ast.walk(audit_loop) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='side')
side_loop.body.insert(0,ast.parse("observed[(row['case_id'],side)]=[]").body[0])
audit_checks={}
audit_scope=dict(np=np,sr=sr,lengths=sa['valid'].sum(axis=0).astype(int),sm=sm,names=names,sa=sa,thresholds=thresholds,record_map={(r['case_id'],r['side']):r for r in p['records']},ref=ref,reconstructed=ref.copy(),cart_expected=ref.copy(),rotate_xyzw=rotate_xyzw,translation=translation,knee_solution=knee_solution,check=lambda k,v:audit_checks.update({k:bool(v)}),observed={})
execute(audit_loop,audit_scope,str(audit_path)+'::read_only_instrumented_loop_only')
audit_records=[dict(case_id=r['case_id'],side=r['side'],frames=audit_scope['observed'][(r['case_id'],r['side'])]) for r in p['records']]
audited_comparison=compare(p['records'],audit_records)
check('frozen_prepare_cartesian_exact',np.array_equal(prepare_scope['cartesian'],refs['cartesian']['reference']))
check('frozen_prepare_ik_exact',np.array_equal(prepare_scope['ik'],refs['two_link_ik']['reference']))
check('original_audit_cartesian_exact',np.array_equal(audit_scope['cart_expected'],refs['cartesian']['reference']))
check('original_audit_ik_exact',np.array_equal(audit_scope['reconstructed'],refs['two_link_ik']['reference']))
actual=[]
for arm in ('cartesian','two_link_ik'):
    x=refs[arm]['reference'];d=x.astype(np.float64)-ref.astype(np.float64);maxshift=float(np.linalg.norm(d,axis=-1).max());maxerr=0.
    unchanged=[j for j in range(24) if j not in [names.index(s+'_'+part) for s in ('L','R') for part in ('Knee','Ankle','Toe')]]
    check(arm+'_prefix61',np.array_equal(x[:61],ref[:61]));check(arm+'_endpose',np.array_equal(x[-1],ref[-1]));check(arm+'_unchanged_anchors',np.array_equal(x[:,:,unchanged],ref[:,:,unchanged]));check(arm+'_unchanged_targets',np.array_equal(refs[arm]['original_goal'],refs['baseline']['original_goal']));check(arm+'_finite',np.isfinite(x).all());check(arm+'_10cm',maxshift<=.1)
    for side in ('L','R'):
        h,k,a,t=[names.index(side+'_'+part) for part in ('Hip','Knee','Ankle','Toe')]
        check(arm+'_'+side+'_same_float32_foot_translation',np.array_equal(x[:,:,[a,t]],prepare_scope['cartesian'][:,:,[a,t]]))
        for u,v in ((h,k),(k,a)):
            er=float(np.abs(np.linalg.norm(x[:,:,u].astype(np.float64)-x[:,:,v].astype(np.float64),axis=-1)-np.linalg.norm(ref[:,:,u].astype(np.float64)-ref[:,:,v].astype(np.float64),axis=-1)).max());maxerr=max(maxerr,er)
            if arm=='two_link_ik':check(side+'_'+names[u]+'_1e5_segment_error',er<=1e-5)
    actual.append(dict(arm=arm,max_joint_shift_m=maxshift,max_segment_error_m=maxerr))
original_other={k:v for k,v in w['checks'].items() if not k.startswith('frozen_proxy_objective_records_')}
check('original_all_non_scalar_checks_pass',all(original_other.values()))
check('all_failures_only_extra_exact_scalar_criterion',len(w['failures'])==117 and all(f.get('check','').startswith('frozen_proxy_objective_records_') for f in w['failures']))
after={n:sha(OUT/n) for n in before};check('original_files_unchanged',before==after)
bindings=dict(original_files=before,preparation_source_sha256=sha(prepare_path),original_audit_source_sha256=sha(audit_path),document_sha256=sha(doc_path),diagnostic_source_sha256=sha(__file__),source_trajectories_sha256=sha(BASE/'raw/val_shared_s11_r0/trajectories.npz'),references={a:sha(OUT/'references'/(a+'.npz')) for a in refs})
report=dict(status='PASS_DOCUMENTED_PREPARATION_INVARIANTS' if all(checks.values()) else 'FAIL_DOCUMENTED_PREPARATION_INVARIANTS',original_witness_status_unchanged='FAIL',independent_witness=True,agent='/root/support_leg_ik_document_follower',time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checks=checks,bindings=bindings,frozen_prepare_loop_recomputation=prepared_comparison,original_follower_loop_recomputation=audited_comparison,documented_coordinate_invariants=actual,documented_contract=dict(geometric_shift_cap_m=.1,segment_error_cap_m=1e-5,prefix_poses=61,endpoint_exact=True,original_goals_exact=True,scalar_record_bitwise_equality_required=False,reason='Document asks geometric bounds, finite arrays, unchanged anchors/goals and source hashes; it contains no exact-equality requirement for logged float64 accepted_shift_m reductions.'),dtypes=dict(baseline=str(ref.dtype),solver_velocity='float64',translation='float64',IK='float64',accepted_shift_norm='float64 -> Python float',stored_reference='float32',prepare_bone_error='float32 norm -> Python float',audit_geometric_error='float64 norm'),array_layout=dict(baseline_strides=list(ref.strides),baseline_c_contiguous=ref.flags.c_contiguous,baseline_f_contiguous=ref.flags.f_contiguous,prepare_copy_strides=list(prepare_scope['refs'].strides)),runtime=dict(python=sys.version,numpy=np.__version__,openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),omp_threads=os.environ.get('OMP_NUM_THREADS')),simulation_calls=0,check_invocations_this_diagnosis=0,run_invocations=0,source_fixes=0,thresholds_changed=False,original_witness_overwritten=False,simulation_authorization=False,efficacy_conclusion=False)
report['interpretation']='Exact scalar equality was an extra audit criterion. Frozen loop recomputations and full saved float32 coordinate equality quantify whether scalar low-bit differences affect any documented invariant. Reduction/copy effects and threading hypotheses must be separated from confirmed observations.'
with (OUT/'SCALAR_CONTRACT_DIAGNOSIS.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
summary={k:report[k] for k in ('status','original_witness_status_unchanged','checks','documented_coordinate_invariants','dtypes','array_layout','runtime','bindings')}
summary['frozen_prepare_scalar_summary']={k:v for k,v in prepared_comparison.items() if k!='all_mismatches'};summary['original_follower_scalar_summary']={k:v for k,v in audited_comparison.items() if k!='all_mismatches'}
lines=['# Separate scalar-contract diagnosis','', 'Status: '+report['status'], '', 'Original DOCUMENT_WITNESS FAIL is untouched. No --check rerun, simulator, gate patch, source modification, threshold change or efficacy conclusion.', '', 'Bitwise equality of logged float64 accepted_shift_m is not required by the frozen document. All numerical evidence and individual scalar discrepancies are retained in JSON.','',json.dumps(summary,indent=2),'','This diagnosis emits no simulator authorization.']
with (OUT/'SCALAR_CONTRACT_DIAGNOSIS.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
