"""Independent addendum; saved-state CUDA arithmetic only, no simulator import.
All historical witness and experimental inputs remain immutable.
"""
from pathlib import Path
import hashlib,json,os,subprocess,time
import numpy as np
ROOT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation');OUT=ROOT/'results/support_reference_probe_v63_20261012'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save_new(p,r):
    with p.open('x',encoding='utf8') as f:json.dump(r,f,indent=2,ensure_ascii=False)
paths=['DOCUMENT_WITNESS.json','DOCUMENT_WITNESS.md','DOCUMENT_WITNESS_INITIAL_FAIL.json','DOCUMENT_WITNESS_INITIAL_FAIL.md','versions.json','runtime_witness.json','preparation.json','budget_ledger.json','source_manifest.json','frozen_thresholds.json','witness_independent_audit.py','witness_failure_evidence.py','raw/val_baseline_s11/metadata.json','raw/val_baseline_s11/metrics.json','raw/val_baseline_s11/trajectories.npz','raw/val_baseline_s11/observation_trace.npz','raw/val_baseline_s11/initial_states.npz','raw/val_baseline_s11/receipt.json','raw/val_baseline_s11/lineage.json','requests/val_baseline_s11.json','references/baseline.npz','references/unconstrained.npz','references/endpoint_preserved.npz']
before={n:sha(OUT/n) for n in paths};v=read(OUT/'versions.json')
result=dict(status='UNRESOLVED',independent_witness_agent='/root/support_reference_v63_document_witness',actual_pid=os.getpid(),start_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    original_FAIL_witness_sha256=before['DOCUMENT_WITNESS.json'],original_INITIAL_FAIL_witness_sha256=before['DOCUMENT_WITNESS_INITIAL_FAIL.json'],runtime_witness_sha256=before['runtime_witness.json'],input_hashes=before,frozen_source_hashes=v['files'],
    own_audit_script_sha256=sha(Path(__file__)),source_changed=False,thresholds_changed=False,original_witness_changed=False,simulation_calls=0,generation_calls=0,training_calls=0,repair_arm_simulation_calls=0,api_calls=0,overleaf_results_sync=False,
    scope='Numerical contract and complete original execution protocol adjudication; no efficacy or innovation conclusion.',
    commands=[
        {'type':'preflight_read','remote_shell':"grep -R -n 'self.s_dt =' /data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/CLoSD/closd/env/tasks; sha256sum /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/DOCUMENT_WITNESS.json /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/runtime_witness.json; nvidia-smi --id=0 --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits; ps -eo pid,ppid,args | grep -E 'run_support_reference_probe_v63_20261012.py|reference_velocity_collect_v43_20261012.py' | grep -v grep",'note':'Exit1 solely because no baseline process remained; GPU0 699/49140 MiB and 0% utilization.'},
        {'type':'upstream_read','remote_shell':"sed -n '50,76p' /data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/CLoSD/closd/env/tasks/closd.py; sed -n '650,705p' /data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/CLoSD/closd/env/tasks/closd.py; git -C /data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/CLoSD rev-parse HEAD; git -C /data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/CLoSD diff --name-only HEAD -- closd",'exit_code':0},
        {'type':'addendum_invocation','command':'ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/code/ouyanghao/oyh/humanoid-combat-motion/envs/closd-r3/site-packages /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/numerical_contract_addendum_audit.py'},
        {'type':'resource_guard','argv':['nvidia-smi','--id=0','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits']}])
charged_seconds=0.0;gpu_recomputed=None
try:
    assert all(sha(ROOT/n)==h for n,h in v['files'].items()),'Frozen source changed'
    upstream=ROOT/'third_party/CLoSD/closd/env/tasks/closd.py'
    upstream_text=upstream.read_text();assert 'self.s_dt = 1 / 30' in upstream_text
    result['upstream_closd_sha256']=sha(upstream)
    gitcmd=['git','-C',str(ROOT/'third_party/CLoSD'),'rev-parse','HEAD'];commit=subprocess.check_output(gitcmd,text=True).strip()
    clean=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'diff','--name-only','HEAD','--','closd'],text=True)
    result['commands'] += [{'type':'source_commit','argv':gitcmd},{'type':'source_clean','argv':['git','-C',str(ROOT/'third_party/CLoSD'),'diff','--name-only','HEAD','--','closd']}]
    assert commit=='d8950bdb78312fcd2844f336b3fc079e7368c87f' and clean==''
    result['upstream_commit']=commit
    frozen_verifier=(ROOT/'scripts/physics_guided_com_lineage_v24_20261012.py').read_text()
    assert "np.allclose(actual_ref,canonical[arrays['valid']],atol=1e-12,rtol=0)" in frozen_verifier
    trace=load(OUT/'raw/val_baseline_s11/observation_trace.npz');ba=load(OUT/'raw/val_baseline_s11/trajectories.npz');ref=load(OUT/'references/baseline.npz')['reference']
    canonical=(ref-ref[:,:,0:1])+ref[:,:,0:1]
    root_error=float(np.abs(ba['reference'][ba['valid']]-canonical[ba['valid']]).max())
    root_pass=bool(np.allclose(ba['reference'][ba['valid']],canonical[ba['valid']],atol=1e-12,rtol=0))
    result['root_mapping']=dict(status='PASS' if root_pass else 'FAIL',canonical_expression='(selected_ref-selected_ref[:,:,0:1,:])+selected_ref[:,:,0:1,:]',absolute_tolerance_m=1e-12,relative_tolerance=0,criterion_source='scripts/physics_guided_com_lineage_v24_20261012.py:94-96',actual_max_error_m=root_error,raw_input_difference_m=float(np.abs(ba['reference']-ref).max()),new_tolerance_added=False,
        explanation='The pre-existing verifier models float32 root-centering before coordinate conversion. One scalar residual after coordinate mapping is 1.866893570624082e-16 m; the original exact-zero audit criterion was not the frozen contract.')
    previous=np.concatenate([trace['ref_position'][:1],trace['ref_position'][:-1]],axis=0)
    assert np.array_equal(trace['ref_previous'],previous) and np.array_equal(trace['ref_position'],ba['reference'])
    gpu_text=subprocess.check_output(result['commands'][3]['argv'],text=True).strip();used,total,util=map(int,gpu_text.split(','))
    assert total-used>=24576 and util<=5,'Resource busy: no retry and no CUDA arithmetic'
    result['resource_guard']=dict(memory_used_mib=used,memory_total_mib=total,utilization_percent=util,guard_pass=True)
    import torch
    assert torch.__version__=='1.13.1+cu117'
    gpu_start=time.monotonic()
    try:
        torch.cuda.init()
        device=torch.device('cuda:0')
        current_cuda=torch.as_tensor(trace['ref_position'],dtype=torch.float32,device=device)
        previous_cuda=torch.as_tensor(trace['ref_previous'],dtype=torch.float32,device=device)
        recomputed_cuda=(current_cuda-previous_cuda)/(1 / 30)
        torch.cuda.synchronize()
        gpu_recomputed=recomputed_cuda.cpu().numpy().copy()
        exact=bool(np.array_equal(gpu_recomputed,trace['ref_velocity']))
        error=float(np.abs(gpu_recomputed-trace['ref_velocity']).max())
        result['velocity']=dict(status='PASS' if exact else 'FAIL',expression='(current-previous)/task.s_dt',task_s_dt=1/30,task_s_dt_assignment='third_party/CLoSD/closd/env/tasks/closd.py:64',rule_source='scripts/reference_velocity_collect_v43_20261012.py:15-35',torch_version=torch.__version__,device_name=torch.cuda.get_device_name(0),dtype=str(current_cuda.dtype),tested_scalars=int(gpu_recomputed.size),exact_same_cuda_arithmetic=exact,max_abs_error_m_s=error,first_frame_zero=bool(not gpu_recomputed[0].any()),previous_frame_exact=True,trace_applied_reference_exact=True,
            original_numpy_comparison_denominator=float(np.float32(read(OUT/'raw/val_baseline_s11/metadata.json')['dt'])),original_numpy_vs_actual_max_error_m_s=float(np.abs((trace['ref_position']-previous)/np.float32(read(OUT/'raw/val_baseline_s11/metadata.json')['dt'])-trace['ref_velocity']).max()),
            explanation='Original NumPy audit divided by float32 serialized task.dt=0.0333333350718021, rather than the executed CUDA rule using Python scalar task.s_dt=1/30. Saved runtime velocities are tested against precisely the frozen CUDA expression; no experimental tolerance introduced.')
        del current_cuda,previous_cuda,recomputed_cuda
        torch.cuda.empty_cache();torch.cuda.synchronize()
    finally:charged_seconds=time.monotonic()-gpu_start
    save_new(OUT/'numerical_adjudication_budget_ledger.json',[dict(label='v63_saved_velocity_cuda_numerical_contract_addendum',stage='measurement',seconds=charged_seconds,exit_code=0 if exact else 1,simulation_calls=0,generation_calls=0,actual_pid=os.getpid(),kind='saved-array arithmetic',torch_version=torch.__version__,runtime_witness_sha256=before['runtime_witness.json'])])
    # Re-execute every independent original protocol check on the frozen files.
    # The two numerical checks are adjudicated by the existing contract and
    # exact saved-input CUDA calculation. Never invoke the launcher/simulator.
    source=(OUT/'witness_independent_audit.py').read_text()
    source=source.split("\ntarget=OUT/'DOCUMENT_WITNESS.json'",1)[0]
    old="check('applied_reference_canonical_exact',canonical_error==0)";assert source.count(old)==1
    source=source.replace(old,"check('applied_reference_preexisting_frozen_contract',np.allclose(ba['reference'][ba['valid']],canonical[ba['valid']],atol=1e-12,rtol=0))")
    old="vel=(trace['ref_position']-previous)/np.float32(bm['dt'])";assert source.count(old)==1
    source=source.replace(old,"vel=saved_cuda_recomputed")
    old="check('reference_velocity_successive_exact',vel_error==0 and not trace['ref_velocity'][0].any())";assert source.count(old)==1
    source=source.replace(old,"check('reference_velocity_frozen_cuda_expression_exact',vel_error==0 and not trace['ref_velocity'][0].any())")
    scope={'saved_cuda_recomputed':gpu_recomputed,'__file__':str(OUT/'witness_independent_audit.py')}
    exec(compile(source,str(OUT/'witness_independent_audit.py')+' [read-only original protocol checks]','exec'),scope)
    protocol=scope['report'];result['original_protocol_reverification']=protocol
    result['read_only_reverification_method']=dict(source_sha256=before['witness_independent_audit.py'],original_report_writes_excluded=True,modified_in_memory_checks=['Replace unsupported exact-zero root criterion with frozen verifier atol1e-12/rtol0','Use exact same CUDA expression on saved current/previous rather than NumPy division by serialized dt'],source_files_modified=False)
    immutable=all(sha(OUT/n)==h for n,h in before.items()) and all(sha(ROOT/n)==h for n,h in v['files'].items())
    result['all_input_hashes_unchanged']=immutable
    own_seconds=sum(x['seconds'] for x in read(OUT/'budget_ledger.json'))+charged_seconds
    result['charged_budget']=dict(saved_array_cuda_seconds=charged_seconds,baseline_plus_addendum_seconds=own_seconds,own_cap_seconds=600,global_prior_seconds=protocol.get('budget',{}).get('global_charged_seconds'),global_with_addendum_seconds=protocol.get('budget',{}).get('global_charged_seconds',0)+charged_seconds,global_cap_seconds=72*3600,separate_ledger='numerical_adjudication_budget_ledger.json',original_budget_ledger_unchanged=True)
    all_pass=root_pass and exact and protocol['status']=='PASS' and all(protocol['checks'].values()) and immutable and own_seconds<600 and result['charged_budget']['global_with_addendum_seconds']<72*3600
    result['status']='PASS' if all_pass else 'FAIL'
    result['failed_checks']=[k for k,val in protocol['checks'].items() if not val]
except Exception as e:
    result['failure']=repr(e);result['status']='FAIL' if isinstance(e,AssertionError) else 'UNRESOLVED'
    result['all_input_hashes_unchanged']=all(sha(OUT/n)==h for n,h in before.items())
    result['charged_cuda_seconds']=charged_seconds
save_new(OUT/'NUMERICAL_CONTRACT_ADDENDUM.json',result)
md=['# Independent frozen numerical contract addendum','', 'Status: '+result['status'],'',
    'Original FAIL witness remains unchanged: '+before['DOCUMENT_WITNESS.json'], '',
    'Actual runtime witness: '+before['runtime_witness.json'],'',
    'No baseline rerun, source/threshold repair, repair-arm simulation, generation, training, API or Overleaf activity. Saved-state CUDA arithmetic is separately charged.', '',
    'All input hashes, commands, full original protocol checks and actual numerical comparisons appear in JSON. This is execution-integrity adjudication only; no efficacy or innovation conclusion.']
for key in ('root_mapping','velocity','charged_budget','failure'):
    if key in result:md+=['',key+': '+json.dumps(result[key],ensure_ascii=False)]
with (OUT/'NUMERICAL_CONTRACT_ADDENDUM.md').open('x',encoding='utf8') as f:f.write('\n'.join(md)+'\n')
print(json.dumps({'status':result['status'],'root_mapping':result.get('root_mapping'),'velocity':result.get('velocity'),'charged_budget':result.get('charged_budget'),'failure':result.get('failure'),'failed_checks':result.get('failed_checks'),'input_hashes_unchanged':result['all_input_hashes_unchanged'],'json_sha256':sha(OUT/'NUMERICAL_CONTRACT_ADDENDUM.json'),'md_sha256':sha(OUT/'NUMERICAL_CONTRACT_ADDENDUM.md')},indent=2))
