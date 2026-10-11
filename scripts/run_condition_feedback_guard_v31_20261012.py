"""C2 simulation protocol; a numeric core check is not simulation efficacy.

New runs require terminal C1 evidence and an independent documented witness.
Frozen runtime, original targets, calibrated metrics and all denominators stay
unchanged. Repetition is one of three paid construction calls, not free data.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

import run_reset_alignment_checks_v4 as runtime

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/condition_feedback_formal_v28_20261012'
OUT = ROOT/'results/condition_feedback_guard_v31_20261012'
DOC = ROOT/'docs/condition_feedback_guard_simulation_v31_20261012.md'
VARIANTS = ('proposed', 'no_agreement', 'no_quality_guard', 'shuffled_agreement')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def prepare():
    state = read(BASE/'queue_state.json')
    assert state['status'] in ('COMPLETE_GATE_NOT_MET',
        'HUMAN_GATE_PASSED_PENDING_VALIDATED_G1_AND_TRAINING_ADAPTERS'), 'Wait for the live C1 job; do not overlap or restart it'
    assert (BASE/'summary.json').is_file() and (BASE/'semantic_diversity_gate.json').is_file()
    # A total joint-quality floor cannot discriminate this selection mechanism.
    summary = read(BASE/'summary.json')['summary']
    assert any(v['joint_pass'] > 0 for v in summary.values()), 'C1 has a total joint-quality floor: diagnose before C2 efficacy runs'
    if OUT.exists():
        version = read(OUT/'versions.json')
        assert all(runtime.sha(ROOT/name) == digest for name, digest in version['files'].items())
        assert runtime.sha(BASE/'summary.json') == version['c1_summary_sha256']
        return
    OUT.mkdir()
    for name in ('source_manifest.json', 'frozen_thresholds.json'):
        shutil.copyfile(BASE/name, OUT/name)
    from physics_guided_round_budget_v34_20261012 import snapshot
    prior = snapshot(ROOT,OUT)
    save(OUT/'prior_budget_snapshot.json', prior)
    version = copy.deepcopy(read(BASE/'versions.json'))
    for name in ('scripts/run_condition_feedback_guard_v31_20261012.py',
        'scripts/physics_guided_run_v31_20261012.sh',
        'scripts/condition_feedback_guard_core_v29_20261012.py',
        'scripts/physics_guided_round_budget_v34_20261012.py',
        'scripts/analyze_condition_feedback_guard_v31_20261012.py', str(DOC.relative_to(ROOT))):
        version['files'][name] = runtime.sha(ROOT/name)
    version.update(role='C2 repeat agreement and quality guard, not established innovation',
        c1_summary_sha256=runtime.sha(BASE/'summary.json'),
        prior_budget_snapshot_sha256=runtime.sha(OUT/'prior_budget_snapshot.json'),
        nominal_candidate_calls_per_arm=3, repeats_charged=True,
        test_inputs_per_arm=432, confirmatory=False, innovation_confirmed=False,
        overleaf_results_sync=False)
    save(OUT/'versions.json', version)
    save(OUT/'budget_ledger.json', [])


def queue_namespace():
    import physics_guided_com_lineage_v24_20261012 as lineage
    import numpy as np
    path = ROOT/'scripts/physics_guided_queue.py'
    code = path.read_text(encoding='utf8').split("if __name__=='__main__':")[0]
    namespace = {'__name__':'guard_v31_queue', '__file__':str(path)}
    exec(compile(code, str(path), 'exec'), namespace)
    for name in ('request', 'certify_cached', 'file_hash', 'motion_hash', 'object_hash'):
        namespace[name] = getattr(lineage, name)
    def certified_with_guards(root,out,path,expected,kwargs,config):
        meta=lineage.certify_cached(root,out,path,expected,kwargs,config)
        n=len(meta['cases']);t=config['episode_steps_30hz']
        assert meta['completion_fence_calls']>=t and 'fetch_results(True)' in meta['completion_fence_protocol']
        assert meta['initial_reset_gate_pass']
        differences=meta['manual_reset_vs_first_action_state_max_difference']
        assert len(differences)==7 and all(v==0 for v in differences.values())
        assert meta['policy_observation_mode']==1 and not meta['sensor_metrics_used_for_selection']
        assert runtime.sha(path/'sensor_diagnostic.npz')==meta['sensor_diagnostic_sha256']
        assert meta['dense_manifold_recording'] is False
        assert not meta['sim_params']['use_gpu_pipeline'] and meta['sim_params']['physx_use_gpu']
        assert all(r['native_device']=='cpu' and r['policy_device'].startswith('cuda')
            for r in meta['bridge_buffers'].values())
        assert len(meta['native_body_properties'])==n
        assert all(len(bodies)==24 and all(len(b['com'])==3 and np.isfinite(b['com']).all()
            for b in bodies) for bodies in meta['native_body_properties'])
        with np.load(path/'trajectories.npz',allow_pickle=False) as arrays:
            assert arrays['valid'].shape==(t,n) and arrays['terminated'].shape==(n,)
            assert np.array_equal(arrays['reference_frame'],np.arange(t))
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
        return meta
    namespace['certify_cached']=certified_with_guards
    namespace['SHELL'] = ROOT/'scripts/physics_guided_run_v31_20261012.sh'
    BaseQueue = namespace['Queue']
    prior = read(OUT/'prior_budget_snapshot.json')
    class Queue(BaseQueue):
        def __init__(self, out):
            self.out=out; self.ledger_path=out/'budget_ledger.json'
            self.ledger=read(self.ledger_path)
        def seconds(self, stage=None):
            return sum(float(r['seconds']) for r in prior['entries']+self.ledger
                if stage is None or r['stage']==stage)
        def preflight(self, stage):
            runtime.OUT=self.out
            runtime.resource_guard()
            assert shutil.disk_usage('/data').free >= 20*1024**3
            assert output_bytes() < 15*1024**3
            assert self.seconds() < 72*3600
            assert self.seconds(stage) < namespace['CONFIG']['budget_gpu_hours'][stage]*3600
    def output_bytes():
        return sum(p.stat().st_size for d in prior['directories']+[str(OUT.relative_to(ROOT))]
            for p in (ROOT/d).rglob('*') if p.is_file())
    # Replace the two inner size polls as well as preflight with cumulative size.
    code_run = code[code.index('    def run('):code.index('\n\n\nclass LoadedArrays')]
    old="sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())"
    assert code_run.count(old)==1
    code_run=code_run.replace(old, 'cumulative_output_bytes()')
    isolated=dict(namespace, cumulative_output_bytes=output_bytes)
    exec(compile('class Runner:\n'+code_run, str(path), 'exec'), isolated)
    Queue.run=isolated['Runner'].run
    namespace['Queue']=Queue
    return namespace


def feedback(namespace, first, repeat, variant, label):
    import numpy as np
    from condition_feedback_guard_core_v29_20261012 import corrected_conditions, agreement
    from physics_guided_core import bounded_update, wrap_angle
    meta, rows, arrays=namespace['load'](first); arrays.close()
    repeated_meta, repeated, arrays=namespace['load'](repeat); arrays.close()
    assert meta['cases']==repeated_meta['cases']
    for key in ('original_goal_sha256','generation_noise_sha256'):
        assert meta[key]==repeated_meta[key]
    config=namespace['CONFIG']
    payload=corrected_conditions(meta, rows, repeated, config,
        use_agreement=variant!='no_agreement')
    if variant=='shuffled_agreement':
        trust=payload['agreement']; shifted=copy.deepcopy(trust)
        mapping=list(range(len(rows)))
        for family in config['families']:
            ids=[i for i,r in enumerate(rows) if r['family']==family]
            for j,i in enumerate(ids):
                donor=ids[(j-3)%len(ids)]
                assert rows[i]['source_id']!=rows[donor]['source_id']
                mapping[i]=donor; shifted[i]=trust[donor]
        errors=np.zeros((len(rows),3)); headings=np.zeros(len(rows))
        for i,(a,b) in enumerate(zip(rows,repeated)):
            # Only labels are shuffled; error magnitudes remain from this input.
            # Invalid local feedback can never be licensed by another source.
            local=agreement(a,b,config['base_thresholds'])
            if local['repeat_completed'] and shifted[i]['position']:
                ea=np.asarray(a.get('endpoint_error_vector')); eb=np.asarray(b.get('endpoint_error_vector'))
                if ea.shape==eb.shape==(3,) and np.isfinite(ea).all() and np.isfinite(eb).all():
                    errors[i]=(ea+eb)/2
            if local['repeat_completed'] and shifted[i]['heading']:
                ha=a.get('heading_error_signed_rad'); hb=b.get('heading_error_signed_rad')
                if ha is not None and hb is not None and np.isfinite([ha,hb]).all():
                    headings[i]=ha+wrap_angle(hb-ha)/2
        payload['proposed_goal']=bounded_update(np.asarray(meta['original_goal'],np.float32),
            np.asarray(meta['proposed_goal'],np.float32),errors,
            config['position_step_m'],config['position_total_m']).astype(np.float32).tolist()
        original=np.asarray(meta['original_heading']); proposed=np.asarray(meta['proposed_heading'])
        h=proposed+np.clip(headings,-np.deg2rad(config['heading_step_deg']),np.deg2rad(config['heading_step_deg']))
        payload['proposed_heading']=(original+np.clip(wrap_angle(h-original),
            -np.deg2rad(config['heading_total_deg']),np.deg2rad(config['heading_total_deg']))).tolist()
        payload['agreement_donor_indices']=mapping
    payload.update(case_ids=[r['case_id'] for r in rows],feedback_source=str(first),
        repeat_source=str(repeat), repeat_outputs={n:runtime.sha(repeat/n)
        for n in ('metadata.json','metrics.json','trajectories.npz')},variant=variant)
    path=OUT/'feedback'/(label+'.json'); path.parent.mkdir(exist_ok=True)
    if path.exists(): assert read(path)==payload
    else: save(path,payload)
    return path


def select_guarded(namespace, paths, label, guard=True):
    import numpy as np
    from condition_feedback_guard_core_v29_20261012 import choose_guarded
    packages=[namespace['load'](p) for p in paths]
    if not guard:
        for _,_,arrays in packages: arrays.close()
        return namespace['select'](OUT,paths,label)
    meta=packages[0][0]
    assert all(m['cases']==meta['cases'] and m['original_goal_sha256']==meta['original_goal_sha256'] for m,_,_ in packages)
    indices,decisions=choose_guarded([rows for _,rows,_ in packages])
    n=len(indices); t=namespace['CONFIG']['episode_steps_30hz']
    refs=np.zeros((t,n,24,3),np.float32); choices=[]
    for i,k in enumerate(indices):
        row=packages[k][1][i]; arrays=packages[k][2]
        chosen=arrays['actual'][arrays['valid'][:,i],i]; assert len(chosen)>0
        length=min(len(chosen),t); refs[:length,i]=chosen[:length]; refs[length:,i]=chosen[length-1]
        choices.append(dict(case_id=row['case_id'],candidate=k,construction_complete=row['completed'],
            path=str(paths[k]),metrics=row,guard_decisions=decisions[i]))
    directory=OUT/'selected';directory.mkdir(exist_ok=True)
    path=directory/(label+'.npz'); selection_path=path.with_suffix('.json')
    selection=dict(choices=choices,source_type='physics_refined',
        original_goal_sha256=meta['original_goal_sha256'],construction_candidates=3)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            assert np.array_equal(old['reference'],refs)
            assert np.array_equal(old['original_goal'],np.asarray(meta['original_goal'],np.float32))
        assert read(selection_path)==selection
    else:
        np.savez_compressed(path,reference=refs,original_goal=np.asarray(meta['original_goal'],np.float32))
        save(selection_path,selection)
    binding=dict(candidate_outputs=[{name:runtime.sha(p/name) for name in
        ('metadata.json','metrics.json','trajectories.npz')} for p in paths],
        selected_motion=namespace['motion_hash'](path),selection_json=runtime.sha(selection_path))
    lineage_path=path.with_suffix('.lineage.json')
    if lineage_path.exists(): assert read(lineage_path)==binding
    else: save(lineage_path,binding)
    for _,_,arrays in packages: arrays.close()
    return path


def main(check,inside):
    if not inside:
        os.execve(str(runtime.PYTHON),[str(runtime.PYTHON),'-u',str(Path(__file__).resolve()),
            '--check' if check else '--run','--inside'],runtime.environment())
    import numpy as np
    import fcntl
    assert np.__version__=='1.23.5'
    prepare(); namespace=queue_namespace()
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=namespace['Queue'](OUT)
        try:
            if not check:
                witness=read(OUT/'runtime_witness.json')
                assert witness['status']=='PASS' and read(OUT/'DOCUMENT_WITNESS_AUDIT.json')['verdict']=='PASS'
            seeds=[11] if check else [11,12,13]
            split='val' if check else 'test'; stage='measurement' if check else 'ablation_g1'
            arms={}
            for seed in seeds:
                first=q.run('%s_native_s%d_r0'%(split,seed),stage,split,'native',seed,0)
                repeat=q.run('%s_repeat_s%d_r1'%(split,seed),stage,split,'native',seed,1,noise_round=0)
                variants=('proposed',) if check else VARIANTS
                for variant in variants:
                    label='%s_%s_s%d'%(split,variant,seed)
                    correction=feedback(namespace,first,repeat,variant,label)
                    candidate=q.run(label+'_r2',stage,split,'native',seed,2,noise_round=0,correction_file=correction)
                    selected=select_guarded(namespace,[first,repeat,candidate],variant+'_s%d'%seed,
                        guard=variant!='no_quality_guard')
                    if not check:
                        replay=q.run('test_eval_%s_s%d'%(variant,seed),stage,split,'replay',seed,replay_file=selected)
                        arms[variant+':'+str(seed)]=str(replay)
                if check:
                    a=read(first/'metadata.json');b=read(repeat/'metadata.json')
                    with np.load(first/'trajectories.npz',allow_pickle=False) as x,np.load(repeat/'trajectories.npz',allow_pickle=False) as y:
                        equal={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
                    # Equality is reported, not imposed; the mechanism diagnoses disagreement.
                    save(OUT/'runtime_witness.json',dict(status='PASS',attempts_per_call=72,
                        calls=3,repeat_array_equal=equal,original_goal_unchanged=a['original_goal_sha256']==b['original_goal_sha256'],
                        paired_noise=a['generation_noise_sha256']==b['generation_noise_sha256'],
                        innovation_confirmed=False,simulation_efficacy_tested=False))
                    q.state('WITNESS_PASS');print('CONDITION_FEEDBACK_GUARD_DOCUMENT_CHECK_PASS',flush=True)
                else: save(OUT/'arms.json',arms)
            if not check:
                # Analysis has a separate review gate: never infer success from receipts.
                q.state('COMPLETE_PENDING_PAIRED_ANALYSIS',innovation_confirmed=False,
                    g1='not_run',downstream='not_run',overleaf_results_sync=False)
        except Exception as error:
            q.state('FAILED_RETAINED',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();main(args.check,args.inside)
