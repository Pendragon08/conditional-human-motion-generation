"""Original-reference human tracking diagnostic after C1, not a new candidate."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil

import run_reset_alignment_checks_v4 as runtime
import run_condition_feedback_guard_v31_20261012 as engine

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/condition_feedback_formal_v28_20261012'
OUT=ROOT/'results/mocap_reference_control_v33_20261012'
DOC=ROOT/'docs/mocap_reference_control_v33_20261012.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')


def prepare():
    status=read(BASE/'queue_state.json')['status']
    assert status in ('COMPLETE_GATE_NOT_MET','HUMAN_GATE_PASSED_PENDING_VALIDATED_G1_AND_TRAINING_ADAPTERS'), 'Wait for live C1; preserve its process'
    assert read(ROOT/'results/contact_com_runtime_v26_20261012/run_receipt.json')['status']=='PASS'
    assert read(ROOT/'results/fixed_reference_com_v27_20261012/replay_summary.json')['status']=='COMPLETE_FIXED_REFERENCE_CALIBRATION'
    if OUT.exists():
        version=read(OUT/'versions.json')
        assert all(runtime.sha(ROOT/name)==digest for name,digest in version['files'].items())
        return
    OUT.mkdir()
    for name in ('source_manifest.json','frozen_thresholds.json'):
        shutil.copyfile(BASE/name,OUT/name)
    from physics_guided_round_budget_v34_20261012 import snapshot
    prior=snapshot(ROOT,OUT)
    save(OUT/'prior_budget_snapshot.json',prior)
    version=copy.deepcopy(read(BASE/'versions.json'))
    for name in ('scripts/mocap_reference_collect_v33_20261012.py',
        'scripts/run_mocap_reference_control_v33_20261012.py',
        'scripts/physics_guided_run_v33_20261012.sh',
        'scripts/physics_guided_round_budget_v34_20261012.py',
        'scripts/run_condition_feedback_guard_v31_20261012.py',str(DOC.relative_to(ROOT))):
        version['files'][name]=runtime.sha(ROOT/name)
    version.update(role='Original MoCap-derived reference tracking control, not innovation efficacy',
        frozen_targets=True,generation_calls=0,diagnostic_only=True,innovation_confirmed=False,
        overleaf_results_sync=False,prior_budget_snapshot_sha256=runtime.sha(OUT/'prior_budget_snapshot.json'))
    save(OUT/'versions.json',version);save(OUT/'budget_ledger.json',[])


def verify_reference(path,namespace):
    import numpy as np
    meta=read(path/'metadata.json');receipt=read(path/'receipt.json')
    assert meta['generation_batch_calls']==0 and meta['diagnostic_only'] is True
    assert receipt['status']=='complete'
    assert runtime.sha(path/'dataset_reference.npz')==meta['dataset_reference_sha256']
    with np.load(path/'dataset_reference.npz',allow_pickle=False) as reference,np.load(path/'trajectories.npz',allow_pickle=False) as replay:
        assert np.array_equal(reference['original_goal'],np.asarray(meta['original_goal'],np.float32))
        selected=reference['reference'];valid=replay['valid']
        canonical=(selected-selected[:,:,0:1,:])+selected[:,:,0:1,:]
        maximum=float(np.max(np.abs(replay['reference'][valid]-canonical[valid])))
        assert np.allclose(replay['reference'][valid],canonical[valid],atol=1e-12,rtol=0)
        assert np.array_equal(replay['reference_frame'],np.arange(120))
        assert reference['reference'].shape==(120,len(meta['cases']),24,3)
    rows=read(path/'metrics.json')
    return dict(attempts=len(rows),**{key:sum(r.get(key,False) for r in rows)
        for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass','terminated')},
        by_offset={str(offset):dict(attempts=sum(r['offset_m']==offset for r in rows),
            **{key:sum(r.get(key,False) for r in rows if r['offset_m']==offset)
            for key in ('completed','condition_pass','physical_pass','tracking_pass','joint_pass','terminated')})
            for offset in (-.15,0.,.15)},
        reference_canonical_max_error_m=maximum,generation_calls=0,
        original_goal_sha256=meta['original_goal_sha256'],dataset_reference_sha256=meta['dataset_reference_sha256'],
        output_hashes={name:runtime.sha(path/name) for name in ('metadata.json','metrics.json','trajectories.npz')})


def main(check,inside):
    if not inside:
        os.execve(str(runtime.PYTHON),[str(runtime.PYTHON),'-u',str(Path(__file__).resolve()),
            '--check' if check else '--run','--inside'],runtime.environment())
    import numpy as np
    import fcntl
    assert np.__version__=='1.23.5'
    prepare();engine.OUT=OUT;namespace=engine.queue_namespace()
    shell=ROOT/'scripts/physics_guided_run_v33_20261012.sh'
    namespace['SHELL']=shell;namespace['Queue'].run.__globals__['SHELL']=shell
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=namespace['Queue'](OUT)
        try:
            if not check:
                assert read(OUT/'runtime_witness.json')['status']=='PASS'
                assert read(OUT/'DOCUMENT_WITNESS_AUDIT.json')['verdict']=='PASS'
            path=q.run('val_mocap_replay_s11','measurement','val','replay',11)
            result=verify_reference(path,namespace)
            assert result['attempts']==72
            if check:
                save(OUT/'runtime_witness.json',dict(status='PASS',measurement=result,
                    innovation_confirmed=False,overleaf_results_sync=False))
                q.state('WITNESS_PASS');print('MOCAP_REFERENCE_CONTROL_DOCUMENT_CHECK_PASS',flush=True)
                return
            test=q.run('test_mocap_replay_s11','reserve','test','replay',11)
            test_result=verify_reference(test,namespace);assert test_result['attempts']==144
            save(OUT/'diagnostic_summary.json',dict(status='COMPLETE_ORIGINAL_REFERENCE_DIAGNOSTIC',
                validation=result,test=test_result,generator_sample_calls=0,thresholds_changed=False,
                all_failures_retained=True,innovation_confirmed=False,overleaf_results_sync=False,
                scope='One fixed-policy original-reference replay; not an independent controller or proof of motion infeasibility'))
            q.state('COMPLETE_ORIGINAL_REFERENCE_DIAGNOSTIC')
        except Exception as error:
            q.state('FAILED_RETAINED',error=repr(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--inside',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();main(args.check,args.inside)
