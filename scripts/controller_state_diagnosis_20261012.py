"""Finite internal-validation engineering diagnosis; no innovation efficacy gate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import run_reset_alignment_checks_v4 as runtime
from verify_execution_lineage_v39_20261012 import read,sha,guard_batch
from physics_guided_com_lineage_v24_20261012 import request

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/support_reference_probe_v63_20261012'
OUT=ROOT/'results/controller_state_diagnosis_20261012'
STATES={
 'multitask':dict(exp_name='CLoSD_multitask_finetune',sha256='7ead3778b95516d8b91b7c5939674e45dd0ad10fcc22da60f9f57d204695f6ac'),
 't2m':dict(exp_name='CLoSD_t2m_finetune',sha256='62577dd2d127888108224d50b7a566eca0fdb650553d8564e268f72a80a2da84'),
 'unfinetuned':dict(exp_name='CLoSD_no_finetune',sha256='a47bc0716ce0c40ffdbf1b093952e07182d47bf645123852df05c8c91faa68c2')}
for state in STATES.values():
    state['path']='third_party/CLoSD/output/CLoSD/'+state['exp_name']+'/Humanoid.pth'
GEN='third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt'
DOC=ROOT/'docs/controller_state_diagnosis_20261012.md'

def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)

def prepare():
    if OUT.exists():raise RuntimeError('Attempt directory exists; never overwrite')
    original=read(PARENT/'versions.json')
    for state in STATES.values():assert sha(ROOT/state['path'])==state['sha256']
    assert sha(ROOT/GEN)==original['generator_weights_sha256']
    source=PARENT/'raw/val_baseline_s11'
    binding=guard_batch(source,read(PARENT/'requests/val_baseline_s11.json'))
    OUT.mkdir();(OUT/'raw').mkdir();(OUT/'requests').mkdir();(OUT/'references').mkdir()
    for name in ('source_manifest.json','frozen_thresholds.json'):
        shutil.copyfile(PARENT/name,OUT/name)
    shutil.copyfile(PARENT/'references/baseline.npz',OUT/'references/fixed.npz')
    files=dict(original['files'])
    for name in ('child_input_guard_20261012.py','reference_child_binding_collect_20261012.py',
                 'controller_state_collect_20261012.py','controller_state_diagnosis_20261012.py'):
        files['scripts/'+name]=sha(ROOT/'scripts'/name)
    files[str(DOC.relative_to(ROOT))]=sha(DOC)
    versions=dict(original,files=files,controller_states=STATES,generator_path=GEN,
        source_manifest_sha256=sha(OUT/'source_manifest.json'),thresholds_sha256=sha(OUT/'frozen_thresholds.json'),
        scope='Three released CLoSD training states; internal validation fixed replay, same algorithm')
    save(OUT/'versions.json',versions)
    meta=read(source/'metadata.json');rows=read(source/'metrics.json')
    assert len(rows)==72 and len({r['source_id'] for r in rows})==24
    save(OUT/'preparation.json',dict(status='PREPARED',baseline_binding=binding,
        reference_sha256=sha(OUT/'references/fixed.npz'),cases=[r['case_id'] for r in rows],
        original_goal_sha256=meta['original_goal_sha256'],seed=11,sources=24,offsets=[-.15,0,.15],
        planned_batches=3,planned_max_seconds=600,analysis_spec='Frozen v23 metrics recomputed from raw; all72 retained; exact exported initial state cross-arm required; no effect claims on failure',
        candidate_efficacy_test=False,innovation_confirmed=False,generation_calls=0))
    (OUT/'budget_ledger.json').write_text('[]',encoding='utf8')
    print('CONTROLLER_STATE_PREPARED',flush=True)

def validate():
    v=read(OUT/'versions.json');p=read(OUT/'preparation.json')
    for name,digest in v['files'].items():assert sha(ROOT/name)==digest,'Source changed: '+name
    for state in v['controller_states'].values():assert sha(ROOT/state['path'])==state['sha256']
    assert sha(ROOT/v['generator_path'])==v['generator_weights_sha256']
    assert sha(OUT/'source_manifest.json')==v['source_manifest_sha256']
    assert sha(OUT/'frozen_thresholds.json')==v['thresholds_sha256']
    assert sha(OUT/'references/fixed.npz')==p['reference_sha256']
    assert subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'rev-parse','HEAD'],text=True).strip()==v['upstream_commit']
    assert subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'diff','--name-only','HEAD','--','closd'],text=True)==''
    return v

def batch(state):
    v=validate();path=OUT/'raw'/state;assert not path.exists(),'Failed/completed invocation retained'
    expected=request(ROOT,OUT,'val','replay',11,0,dict(noise_round=0,replay_file=OUT/'references/fixed.npz'))
    expected.update(controller_state=state,controller_weights_sha256=v['controller_states'][state]['sha256'])
    wanted=OUT/'requests'/(state+'.json');save(wanted,expected)
    prior=read(sorted((ROOT/'results/round_resource_status_20261012').glob('snapshot_*.json'))[-1])
    dirs=[ROOT/n for n in prior['directories']]
    if OUT not in dirs:dirs.append(OUT)
    rows=[r for d in dirs if (d/'budget_ledger.json').exists() for r in read(d/'budget_ledger.json')]
    own=read(OUT/'budget_ledger.json')
    timeout=min(240,600-sum(r['seconds'] for r in own),72*3600-sum(r['seconds'] for r in rows),
        4*3600-sum(r['seconds'] for r in rows if r['stage']=='measurement'))
    assert timeout>=60 and shutil.disk_usage('/data').free>=20*1024**3
    assert sum(p.stat().st_size for d in dirs for p in d.rglob('*') if p.is_file())+prior['archive_bytes']<15*1024**3
    runtime.OUT=OUT;resources=runtime.resource_guard()
    cmd=[str(runtime.PYTHON),'-u',str(ROOT/'scripts/controller_state_collect_20261012.py'),
        '--manifest',str(OUT/'source_manifest.json'),'--split','val','--mode','replay','--seed','11','--round','0',
        '--noise-round','0','--replay-file',str(OUT/'references/fixed.npz'),'--output',str(path),
        '--thresholds',str(OUT/'frozen_thresholds.json'),'--request-file',str(wanted),
        '--execution-versions',str(OUT/'versions.json'),'--controller-state',state]
    (OUT/'state.json').write_text(json.dumps(dict(status='RUNNING',arm=state,pid=os.getpid(),resources=resources)),encoding='utf8')
    started=time.monotonic();code=-1
    try:
        with (OUT/(state+'.log')).open('x') as log:
            code=subprocess.run(cmd,cwd=ROOT/'third_party/CLoSD',env=runtime.environment(),stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
    finally:
        own.append(dict(label=state,stage='measurement',seconds=time.monotonic()-started,exit_code=code,request_sha256=sha(wanted)))
        (OUT/'budget_ledger.json').write_text(json.dumps(own,indent=2),encoding='utf8')
        save(OUT/(state+'_parent_return.json'),dict(child_returncode=code,pid=os.getpid(),time=time.time(),command=cmd,
            note='subprocess return observation; parent program terminal status is separate'))
    assert code==0,'Child failed; retained and charged'
    binding=guard_batch(path,expected);loaded=read(path/'loaded_checkpoint_binding.json');inputs=read(path/'loaded_input_binding.json')
    assert loaded['controller_state']==state and loaded['request_sha256']==sha(wanted)
    assert loaded['versions_sha256']==inputs['versions_sha256']==sha(OUT/'versions.json')
    assert inputs['loaded_replay_content_sha256']==expected['replay_motion']
    assert sorted(loaded['actual_loaded_weight_bytes'].values())==sorted([v['controller_states'][state]['sha256'],v['generator_weights_sha256']])
    meta=read(path/'metadata.json');scores=read(path/'metrics.json');p=read(OUT/'preparation.json')
    assert meta['generation_batch_calls']==0 and len(scores)==72
    assert meta['original_goal_sha256']==p['original_goal_sha256']
    assert [r['case_id'] for r in scores]==p['cases']
    assert meta['initial_reset_gate_pass'] and all(x==0 for x in meta['initial_rigid_cache_consistency'].values())
    return dict(binding=binding,counts={key:sum(int(r[key]) for r in scores) for key in
        ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},cases=72,
        loaded_checkpoint_binding_sha256=sha(path/'loaded_checkpoint_binding.json'),loaded_input_binding_sha256=sha(path/'loaded_input_binding.json'))

def analyze():
    from physics_guided_metric_v23_20261012 import metrics
    from metric_numeric_equivalence import compare_metrics
    limits=read(OUT/'frozen_thresholds.json')['thresholds'];base_initial=None;base_meta=None;reports={};bindings={}
    numeric=('endpoint_error_m','heading_error_rad','surface_slip_proxy_fraction','penetration_fraction','jerk_p95_m_s3','tracking_mean_m')
    for state in STATES:
        path=OUT/'raw'/state;guard_batch(path,read(OUT/'requests'/(state+'.json')))
        meta=read(path/'metadata.json');saved=read(path/'metrics.json')
        with np.load(path/'trajectories.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        computed=[metrics(arrays,i,meta,limits) for i in range(72)]
        equivalence=compare_metrics(computed,saved)
        with np.load(path/'initial_states.npz',allow_pickle=False) as z:initial={k:z[k] for k in z.files}
        fields={'_humanoid_root_states','_dof_pos','_dof_vel','_rigid_body_pos','_rigid_body_rot','_rigid_body_vel','_rigid_body_ang_vel'}
        assert all(np.array_equal(initial['bootstrap_'+k],initial['first_action_'+k]) for k in fields)
        if base_initial is None:base_initial=initial;base_meta=meta
        assert set(initial)==set(base_initial) and all(np.array_equal(initial[k],base_initial[k]) for k in initial),'Exported initial state differs; stop attribution'
        assert all(meta[k]==base_meta[k] for k in ('cases','original_goal','original_heading'))
        with np.load(OUT/'references/fixed.npz',allow_pickle=False) as ref:
            canonical=(ref['reference']-ref['reference'][:,:,0:1,:])+ref['reference'][:,:,0:1,:]
            assert np.allclose(arrays['reference'][arrays['valid']],canonical[arrays['valid']],atol=1e-12,rtol=0)
        means={};denom={}
        for key in numeric:
            values=[r[key] for r in computed if r.get(key) is not None and np.isfinite(r[key])]
            means[key]=float(np.mean(values)) if values else None;denom[key]=dict(finite=len(values),total=72)
        reports[state]=dict(counts={k:sum(int(r[k]) for r in computed) for k in
            ('completed','condition_pass','physical_pass','tracking_pass','joint_pass')},means=means,denominators=denom,metric_equivalence=equivalence)
        for name in ('trajectories.npz','initial_states.npz','metadata.json','metrics.json','lineage.json','receipt.json','loaded_checkpoint_binding.json','loaded_input_binding.json'):
            bindings[str((path/name).relative_to(ROOT))]=sha(path/name)
    result=dict(status='COMPLETE_CHECKPOINT_STATE_DIAGNOSIS',reports=reports,input_hashes=bindings,
        same_algorithm_different_training_states=True,exported_initial_states_exact=True,hidden_physx_identity_established=False,
        sources=24,seed=11,inputs_per_state=72,generation_calls=0,simulator_batches=3,
        candidate_efficacy_test=False,confirmed_innovations=0,g1_or_downstream_run=False,overleaf_results_sync=False,
        scope='Internal validation, upstream train/test mix, simulated human; surface-slip/contact and tracking proxies. Not a new algorithm, innovation efficacy or robot utility result.')
    save(OUT/'analysis.json',result)
    print(json.dumps({s:r['counts'] for s,r in reports.items()},indent=2),flush=True)
    print('CONTROLLER_STATE_DIAGNOSIS_COMPLETE',flush=True)

def main(mode):
    if mode=='prepare':prepare();return
    validate()
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if mode=='check':
                report=batch('multitask')
                save(OUT/'runtime_witness.json',dict(status='PASS',report=report,versions_sha256=sha(OUT/'versions.json')))
                print('CONTROLLER_STATE_DOCUMENT_CHECK_PASS',flush=True)
            else:
                witness=read(OUT/'DOCUMENT_WITNESS.json')
                assert witness['status']=='PASS' and witness['runtime_witness_sha256']==sha(OUT/'runtime_witness.json')
                assert witness['versions_sha256']==sha(OUT/'versions.json')
                for state in ('t2m','unfinetuned'):batch(state)
                analyze()
            (OUT/'state.json').write_text(json.dumps(dict(status='CHECK_COMPLETE' if mode=='check' else 'COMPLETE_CHECKPOINT_STATE_DIAGNOSIS')),encoding='utf8')
        except Exception as e:
            (OUT/'state.json').write_text(json.dumps(dict(status='FAILED_RETAINED',error=repr(e))),encoding='utf8');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','check','run'):g.add_argument('--'+mode,action='store_true')
    args=p.parse_args();main(next(m for m in ('prepare','check','run') if getattr(args,m)))
