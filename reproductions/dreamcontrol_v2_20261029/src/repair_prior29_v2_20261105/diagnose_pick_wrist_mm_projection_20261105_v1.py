"""Refine only the residual wrist position of one Pick fit, then independently replay."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
OUT=S/'pick_wrist_mm_projection_v1'
CANDIDATE=10000007
SOURCE=S/'pick_legacy_seed_diagnostic_v4/candidate_10000007'
PLAN=S/'generation_pipeline_v5_launch/config.json'
PLAN_SHA='5d2cfca7b6da4cc9f6b47fb36c63db05ddb6cf8f26244a8189f2486b7a81403a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def refine(plan, p):
    import postopt_v2 as solver_source

    report_path=SOURCE/'report.json'
    assert sha(report_path)=='80bb37f4ae16d3b7187f8b809b799ee811e85ff0f93be719a576b5f211f2413d'
    original=read(report_path)
    assert original['candidate_id']==CANDIDATE and original['status']=='generated_candidate_rejected'
    assert original['source_sample_sha256']=='ee319cdb93544bc39d404d25178a5b6e62cba9f6af3312e439574e0c284ebf2c'
    branch=original['stages'][2]
    assert [k for k,v in branch['checks'].items() if not v]==['hand_target']
    a=SOURCE/'stage_a_decoded_realization.npz'
    b=SOURCE/'stage_b_03_raw.npz'
    assert sha(a)==original['stage_a_sha256'] and sha(b)==branch['sha256']
    raw=Path(original['source_sample'])
    sample=solver_source.load_sample(raw, original['source_sample_sha256'])
    assert sample['frames']==120 and sample['wrist_mask'][:,1].all() and not sample['wrist_mask'][:,0].any()
    solver=solver_source.MotionPreservingSolverV2(p.bound(plan['urdf']), sample['frames'], p.bound(plan['fk']))
    with np.load(a,allow_pickle=False) as data:
        qa, ra=data['q'],data['root']
    with np.load(b,allow_pickle=False) as data:
        qb, rb=data['q'],data['root']
    qa, pa, _, _=solver.cpu_state(qa, ra)
    stage_a={'q':qa,'root':ra,'points':pa[:,solver.kp]}
    before, _, xyz, _=solver.measure(sample,qb,rb,stage_a)
    checks_before=solver_source.acceptance_checks(before,solver.policy)
    assert checks_before==branch['checks'] and abs(before['wrist_max_l2_m']-branch['metrics']['wrist_max_l2_m'])<1e-8
    error=sample['targets_world'][:,21]-xyz[:,21]
    norms=np.linalg.norm(error,axis=1)
    # Euclidean projection to the interior of the unchanged 3 cm wrist ball.
    limit=solver.policy.wrist_position_max_m-.00005
    scale=np.divide(np.maximum(norms-limit,0.),norms,out=np.zeros_like(norms),where=norms>0.)
    delta=scale[:,None]*error
    assert np.linalg.norm(delta,axis=1).max()<.004
    corrected=rb.copy()
    corrected[:,4:]+=delta.astype(corrected.dtype)
    metrics,q,points,wquat=solver.measure(sample,qb,corrected,stage_a)
    checks=solver_source.acceptance_checks(metrics,solver.policy)
    output=OUT/'postopt'/f'candidate_{CANDIDATE}'
    output.mkdir(parents=True)
    updated={k:v for k,v in original.items() if k not in ('status','rejection_reason','stages','metrics','checks')}
    updated.update(status='kinematic_candidate_ready_for_native_probe' if all(checks.values()) else 'generated_candidate_rejected',
                   refinement='minimum norm root translation for final 3cm wrist position ball; unchanged goals and all final gates rechecked',
                   source_baseline_report_sha256=sha(report_path), source_baseline_state_sha256=sha(b),
                   stage_a_reference_state_sha256=sha(a), base_metrics=before, metrics=metrics, checks=checks,
                   max_added_root_translation_m=float(np.linalg.norm(delta,axis=1).max()),
                   translated_frames=int(np.count_nonzero(scale)),
                   passed_requested_kinematic_checks=all(checks.values()),
                   max_hand_error_m=metrics['wrist_max_l2_m'],max_wrist_angle_rad=metrics['wrist_max_angle_rad'],
                   max_joint_projection_rad=metrics['joint_projection_rad'],
                   native_scene_collision_replay='pending', eligible_for_calibrated_scene_reference=False)
    if all(checks.values()):
        destination=output/'robot29.npz'
        with destination.open('xb') as stream:
            np.savez_compressed(stream,q=q,root_pos=corrected[:,4:],root_quat=corrected[:,:4],
                keypoints=points.astype(np.float32),left_wrist_quat=wquat[:,0].astype(np.float32),
                right_wrist_quat=wquat[:,1].astype(np.float32),hand_closed=sample['hand_closed'],
                planned_object_positions=sample['desired_objects'],desired_wrist_quat_wxyz=sample['orientations'],
                joint_names=np.array(solver.names),fps=np.array(20.),sample_times=np.arange(120)/20.,
                task_object_position=np.array(sample['metadata']['task_object_position']),
                task_object_quaternion_wxyz=np.array([1.,0.,0.,0.]),
                metadata_utf8=np.frombuffer(json.dumps(sample['metadata']).encode(),dtype=np.uint8))
        updated.update(output=str(destination),output_sha256=sha(destination))
    else:
        updated['rejection_reason']='root_wrist_projection_failed_unchanged_gates'
    p.save(output/'report.json',updated,exclusive=True)
    return updated


def main():
    assert sha(PLAN)==PLAN_SHA
    plan=read(PLAN)
    sys.path.insert(0,str(Path(plan['sources']['controller']['path']).parent))
    import pipeline_controller_v2 as p

    cfg=p.load_config(p.bound(plan['training_config']))
    assert sha(Path(p.__file__))==plan['sources']['controller']['sha256']
    p.bound(plan['sources']['postopt'])
    OUT.mkdir()
    p.resource_guard(p.resource_snapshot(Path(cfg['project_root'])),plan['cohort_write_budget_bytes'])
    p.save(OUT/'status.json',{'status':'cpu_rechecking_unchanged_gates','candidate_id':CANDIDATE})
    try:
        report=refine(plan,p)
        if not report['passed_requested_kinematic_checks']:
            p.save(OUT/'status.json',{'status':report['status'],'candidate_id':CANDIDATE,
                 'failed_checks':[k for k,v in report['checks'].items() if not v],'admitted_references':0})
            return
        cohort=S/'new_references_v5/pick/cohort_0007'
        rows=p.producer_records(cohort/'raw','pick',[CANDIDATE],report['checkpoint_sha256'])
        staged=p.stage_native(plan,'pick',OUT/'postopt',OUT/'native_input',rows)
        assert len(staged)==1
        native=read(p.bound(plan['native_runtime']))
        script=p.bound(plan['sources']['native_pick'])
        env=p.runtime_env(native['env'],remove_visibility=True)
        env['PYTHONPATH']=str(script.parent)+os.pathsep+env.get('PYTHONPATH','')
        result=OUT/'native'
        argv=[native['python'],'-B','-u',str(script),'--headless','--device',plan['native_device'],
              '--output-dir',str(result),'--candidate-dir',str(staged[0][1])]
        p.phase(PLAN,PLAN_SHA,plan,cfg,OUT,OUT,'native',argv,env,native['cwd'],(result/'report.json','native_replay_completed'))
        p.native_proofs('pick',staged,result)
        row=rows[0]
        p.save(OUT/'result.json',{'status':'one_pick_native_probe_complete','candidate_id':CANDIDATE,
               'new_samples':0,'admitted_references':int(row['admitted']),'records':rows},exclusive=True)
        p.save(OUT/'status.json',{'status':'one_pick_native_probe_complete','candidate_id':CANDIDATE,
               'admitted_references':int(row['admitted']),'result_sha256':sha(OUT/'result.json')})
    except BaseException:
        (OUT/'traceback.txt').write_text(traceback.format_exc())
        p.save(OUT/'status.json',{'status':'failed_preserved','candidate_id':CANDIDATE,'admitted_references':0})
        raise


if __name__=='__main__':
    main()
