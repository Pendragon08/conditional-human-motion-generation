"""Resumable, budget-limited paired experiment. No success-only collection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from physics_guided_core import bounded_update, rank_candidate, wrap_angle
from physics_guided_lineage import request, certify_cached, file_hash, motion_hash, object_hash

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/physics_guided_20261011.json').read_text())
SHELL=ROOT/'scripts/physics_guided_run.sh'


class Queue:
    def __init__(self,out):
        self.out=out
        self.ledger_path=out/'budget_ledger.json'
        self.ledger=json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []
        if not self.ledger:
            self.ledger.append(dict(stage='measurement',label='initial_development_and_doc_checks',
                seconds=1800,status='conservative_reserved_debit',note='Includes early development failures and independent witness'))
            self.save_ledger()
    def save_ledger(self):
        self.ledger_path.write_text(json.dumps(self.ledger,indent=2))
    def state(self,status,**values):
        content=dict(status=status,time=time.time(),**values)
        (self.out/'queue_state.json').write_text(json.dumps(content,indent=2))
        print('PHYSICS_GUIDED_QUEUE',json.dumps(content),flush=True)
    def seconds(self,stage=None):
        return sum(r['seconds'] for r in self.ledger if stage is None or r['stage']==stage)
    def preflight(self,stage):
        gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert int(gpu[0])<2048 and int(gpu[1])<=5, 'GPU0 has another active workload; do not displace it'
        assert shutil.disk_usage('/data').free>=CONFIG['min_data_free_bytes'], 'Data free space below20GiB'
        size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        assert size<CONFIG['max_output_bytes'], 'Experiment output limit exceeded'
        assert self.seconds()<72*3600, 'Global GPU budget exhausted'
        assert self.seconds(stage)<CONFIG['budget_gpu_hours'][stage]*3600, 'Stage budget exhausted'
    def run(self,label,stage,split,mode,seed,round_id=0,**kwargs):
        path=self.out/'raw'/label
        expected=request(ROOT,self.out,split,mode,seed,round_id,kwargs)
        if (path/'receipt.json').exists():
            certify_cached(ROOT,self.out,path,expected,kwargs,CONFIG)
            assert any(r.get('label')==label for r in self.ledger), 'Missing resumed task cost; reconcile before resuming'
            return path
        if path.exists():
            raise RuntimeError('Incomplete prior attempt retained; explicit fix/version needed: '+str(path))
        self.preflight(stage)
        request_dir=self.out/'requests'; request_dir.mkdir(exist_ok=True)
        request_path=request_dir/(label+'.json')
        request_path.write_text(json.dumps(expected,indent=2))
        args=['bash',str(SHELL),'collect','--output',str(path),'--split',split,'--mode',mode,
              '--seed',str(seed),'--round',str(round_id),'--thresholds',str(self.out/'frozen_thresholds.json'),
              '--request-file',str(request_path)]
        for key,value in kwargs.items():
            args.extend(['--'+key.replace('_','-'),str(value)])
        log=self.out/(label+'.log'); start=time.monotonic()
        remaining=min(72*3600-self.seconds(),CONFIG['budget_gpu_hours'][stage]*3600-self.seconds(stage))
        self.state('RUNNING',label=label,stage=stage,split=split)
        with log.open('w') as stream:
            process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            reason=None
            while process.poll() is None:
                if time.monotonic()-start>=remaining:
                    reason='budget_exhausted'
                elif shutil.disk_usage('/data').free<CONFIG['min_data_free_bytes']:
                    reason='data_space_gate'
                elif sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())>CONFIG['max_output_bytes']:
                    reason='output_size_gate'
                if reason:
                    os.killpg(process.pid,15)
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,9); process.wait()
                    break
                time.sleep(10)
        elapsed=time.monotonic()-start
        self.ledger.append(dict(stage=stage,label=label,seconds=elapsed,exit_code=process.returncode,stop_reason=reason))
        self.save_ledger()
        if process.returncode!=0 or reason or not (path/'receipt.json').exists():
            raise RuntimeError('Failed stage '+label+'; inspect '+str(log))
        certify_cached(ROOT,self.out,path,expected,kwargs,CONFIG)
        return path


class LoadedArrays(dict):
    def close(self):
        self.clear()


def load(path):
    with np.load(path/'trajectories.npz',allow_pickle=False) as archive:
        arrays=LoadedArrays({key:archive[key] for key in archive.files})
    return (json.loads((path/'metadata.json').read_text()),
            json.loads((path/'metrics.json').read_text()),
            arrays)


def correction(out,previous,ordinal,shuffled=False,no_feedback=False):
    meta,rows,_=load(previous)
    original=np.asarray(meta['original_goal'],dtype=np.float32)
    proposed=np.asarray(meta['proposed_goal'],dtype=np.float32)
    errors=np.array([r.get('endpoint_error_vector',[0,0,0]) if r['length']>60 else [0,0,0] for r in rows])
    headings=np.array([r.get('heading_error_signed_rad',0) if r['length']>60 else 0 for r in rows])
    if shuffled:
        # Fixed within-family derangement; no change to cases or evaluation goals.
        for family in CONFIG['families']:
            ids=[i for i,r in enumerate(rows) if r['family']==family]
            # Each source has three offsets. Shift by a whole source so every
            # recipient receives another source's feedback at the same offset.
            assert all(rows[ids[j]]['source_id']!=rows[ids[(j-3)%len(ids)]]['source_id'] for j in range(len(ids)))
            errors[ids]=np.roll(errors[ids],3,axis=0); headings[ids]=np.roll(headings[ids],3)
    if no_feedback:
        errors[:]=0; headings[:]=0
    next_goal=bounded_update(original,proposed,errors,CONFIG['position_step_m'],CONFIG['position_total_m']).astype(np.float32)
    original_h=np.asarray(meta['original_heading'])
    proposed_h=np.asarray(meta['proposed_heading'])
    step=np.deg2rad(CONFIG['heading_step_deg']); total=np.deg2rad(CONFIG['heading_total_deg'])
    next_h=proposed_h+np.clip(headings,-step,step)
    next_h=original_h+np.clip(wrap_angle(next_h-original_h),-total,total)
    path=out/'feedback'/('feedback_'+ordinal+'.json'); path.parent.mkdir(exist_ok=True)
    payload=dict(case_ids=[r['case_id'] for r in meta['cases']],original_goal_sha256=meta['original_goal_sha256'],
                 proposed_goal=next_goal.tolist(),proposed_heading=next_h.tolist(),
                 feedback_source=str(previous),shuffled=shuffled,no_feedback=no_feedback)
    if path.exists(): assert object_hash(json.loads(path.read_text()))==object_hash(payload), 'Feedback input changed'
    else: path.write_text(json.dumps(payload,indent=2))
    return path


def select(out,paths,label,raw=False,tracking_only=False):
    packages=[load(p) for p in paths]
    meta=packages[0][0]
    assert all(m['original_goal_sha256']==meta['original_goal_sha256'] for m,_,_ in packages)
    n=len(meta['cases']); t=CONFIG['episode_steps_30hz']
    refs=np.zeros((t,n,24,3),np.float32)
    choices=[]
    for i in range(n):
        k=min(range(len(paths)),key=lambda j:rank_candidate(packages[j][1][i],CONFIG['base_thresholds'],tracking_only))
        row=packages[k][1][i]; arrays=packages[k][2]
        valid=arrays['valid'][:,i]; chosen=arrays['reference' if raw else 'actual'][valid,i]
        assert len(chosen)>0
        length=min(len(chosen),t)
        refs[:length,i]=chosen[:length]; refs[length:,i]=chosen[length-1]
        choices.append(dict(case_id=row['case_id'],candidate=k,construction_complete=row['completed'],
                            path=str(paths[k]),metrics=row))
    dest=out/'selected'; dest.mkdir(exist_ok=True)
    path=dest/(label+'.npz')
    selection=dict(choices=choices,source_type='raw' if raw else 'physics_refined',
        original_goal_sha256=meta['original_goal_sha256'],construction_candidates=len(paths))
    selected_json=dest/(label+'.json')
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            assert np.array_equal(old['reference'],refs), 'Previously selected motion changed'
            assert np.array_equal(old['original_goal'],np.array(meta['original_goal'],np.float32))
        assert object_hash(json.loads(selected_json.read_text()))==object_hash(selection), 'Previously selected candidate changed'
    else:
        np.savez_compressed(path,reference=refs,original_goal=np.array(meta['original_goal'],dtype=np.float32))
        selected_json.write_text(json.dumps(selection,indent=2))
    lineage=dict(candidate_outputs=[{name:file_hash(p/name) for name in ('metadata.json','metrics.json','trajectories.npz')}
                                   for p in paths],selected_motion=motion_hash(path),selection_json=file_hash(selected_json))
    lineage_path=dest/(label+'.lineage.json')
    if lineage_path.exists(): assert json.loads(lineage_path.read_text())==lineage
    else: lineage_path.write_text(json.dumps(lineage,indent=2))
    for _,_,arrays in packages: arrays.close()
    return path


def main(out):
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    with (out/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        q=Queue(out)
        try:
            witness=json.loads((out/'witness_doc_v4/receipt.json').read_text())
            assert witness['status']=='complete' and witness['cases']==6
            with np.load(out/'witness_doc_v4/trajectories.npz',allow_pickle=False) as fixture:
                assert np.array_equal(fixture['reference_frame'],np.arange(CONFIG['episode_steps_30hz']))
            frozen=out/'frozen_thresholds.json'
            if not frozen.exists():
                # Validation measurement is recorded before test. Keep the
                # predeclared SI thresholds; no post hoc quantile relaxation.
                frozen.write_text(json.dumps(dict(thresholds=CONFIG['base_thresholds'],
                    calibration='predeclared conservative proxy thresholds; inspect validation validity before test',
                    config_sha256=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()),indent=2))
            assert json.loads(frozen.read_text())['config_sha256']==hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
            assert json.loads(frozen.read_text())['thresholds']==CONFIG['base_thresholds']
            versions=json.loads((out/'versions.json').read_text())
            for name,digest in versions['files'].items():
                assert file_hash(ROOT/name)==digest, 'Versioned implementation/input changed: '+name
            manifest=json.loads((out/'source_manifest.json').read_text())
            snapshot=Path(manifest['snapshot'])
            assert snapshot.name==versions['assets_revision']==CONFIG['assets_revision']
            weights={
                'generator_weights_sha256':ROOT/'third_party/CLoSD/closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt',
                'controller_weights_sha256':ROOT/'third_party/CLoSD/output/CLoSD/CLoSD_multitask_finetune/Humanoid.pth',
                'text_motion_evaluator_weights_sha256':snapshot/'evaluators/t2m/text_mot_match/model/finest.tar'}
            for name,path in weights.items(): assert file_hash(path)==versions[name], 'Frozen weight changed'
            for name,digest in versions['external_motion_inputs'].items(): assert file_hash(name)==digest
            for name,digest in versions['external_semantic_inputs'].items(): assert file_hash(name)==digest
            # Run two prefixes in document witness, then all24 validation sources.
            v=q.run('val_planned_s11_r0','measurement','val','planned',11)
            native_v=q.run('val_native_s11_r0','measurement','val','native',11)
            for task in (v,native_v):
                meta,rows,arrays=load(task)
                assert meta['body_coordinate_roundtrip_max_m']<1e-5
                assert meta['target_world_roundtrip_max_m']<1e-4
                assert len(rows)==72
                assert any(r['length']>60 for r in rows), 'No examples reached generated portion'
                assert any(r.get('support_observations',0)>0 for r in rows), 'Contact geometry has no support observations'
                for family in CONFIG['families']:
                    family_rows=[r for r in rows if r['family']==family]
                    assert len(family_rows)==36 and any(r['length']>60 for r in family_rows)
                    assert any(r['completed'] for r in family_rows), 'No full validation episode in family'
                    assert all(r.get('support_observations',0)>0 for r in family_rows if r['completed'])
                arrays.close()
            (out/'measurement_gate.json').write_text(json.dumps(dict(status='pass',validation_cases=144,
                joint_thresholds_frozen=True,threshold_policy='predeclared SI limits retained without test-dependent adjustment',
                protocol_deviation='Threshold values frozen before validation; validation checks measurement validity rather than estimating cutoffs',
                quality_discrimination_warning='Inspect per-component rates; a zero joint-pass validation rate is a possible floor effect',
                semantic_gate='independent evaluation required before robot/downstream claims'),indent=2))
            arms={}
            for seed in CONFIG['seeds']:
                planned=[q.run('test_planned_s%d_r%d'%(seed,r),'comparison','test','planned',seed,r) for r in range(3)]
                native=[q.run('test_native_s%d_r%d'%(seed,r),'comparison','test','native',seed,r) for r in range(3)]
                proposed=[native[0]]
                for r in (1,2):
                    feedback=correction(out,proposed[-1],'proposed_s%d_r%d'%(seed,r))
                    proposed.append(q.run('test_proposed_s%d_r%d'%(seed,r),'comparison','test','native',seed,r,
                                          noise_round=0,correction_file=feedback))
                comparisons={
                    'raw':select(out,planned[:1],'raw_s%d'%seed,raw=True),
                    'ordinary':select(out,planned[:1],'ordinary_s%d'%seed),
                    'random_best':select(out,planned,'random_best_s%d'%seed),
                    'native_single':select(out,native[:1],'native_single_s%d'%seed),
                    'native_best':select(out,native,'native_best_s%d'%seed),
                    'proposed':select(out,proposed,'proposed_s%d'%seed)}
                for name,path in comparisons.items():
                    replay=q.run('test_eval_%s_s%d'%(name,seed),'comparison','test','replay',seed,replay_file=path)
                    arms[name+':'+str(seed)]=str(replay)
                for ablation in ('no_feedback','shuffled','planned_prefix'):
                    candidates=[planned[0] if ablation=='planned_prefix' else native[0]]
                    for r in (1,2):
                        feedback=correction(out,candidates[-1],'%s_s%d_r%d'%(ablation,seed,r),
                                            shuffled=ablation=='shuffled',no_feedback=ablation=='no_feedback')
                        candidates.append(q.run('test_%s_s%d_r%d'%(ablation,seed,r),'ablation_g1','test',
                                                'planned' if ablation=='planned_prefix' else 'native',seed,r,
                                                noise_round=0,correction_file=feedback))
                    selected=select(out,candidates,'%s_s%d'%(ablation,seed))
                    replay=q.run('test_eval_%s_s%d'%(ablation,seed),'ablation_g1','test','replay',seed,replay_file=selected)
                    arms[ablation+':'+str(seed)]=str(replay)
                selected=select(out,proposed,'tracking_only_s%d'%seed,tracking_only=True)
                replay=q.run('test_eval_tracking_only_s%d'%seed,'ablation_g1','test','replay',seed,replay_file=selected)
                arms['tracking_only:'+str(seed)]=str(replay)
                (out/'arms.json').write_text(json.dumps(arms,indent=2))
            subprocess.run([os.sys.executable,str(ROOT/'scripts/physics_guided_analyze.py'),'--output',str(out)],check=True)
            q.preflight('comparison')
            semantic_start=time.monotonic()
            remaining=min(600.,72*3600-q.seconds(),CONFIG['budget_gpu_hours']['comparison']*3600-q.seconds('comparison'))
            try:
                with (out/'semantic_proxy.log').open('w') as stream:
                    result=subprocess.run([os.sys.executable,str(ROOT/'scripts/physics_guided_semantics.py'),
                        '--output',str(out)],stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
                if result.returncode:
                    (out/'semantic_diversity_gate.json').write_text(json.dumps(dict(status='unavailable',
                        reason='Released evaluator invocation failed; see semantic_proxy.log')))
            except subprocess.TimeoutExpired:
                (out/'semantic_diversity_gate.json').write_text(json.dumps(dict(status='unavailable',
                    reason='Semantic evaluator exhausted its time budget')))
            finally:
                q.ledger.append(dict(stage='comparison',label='independent_semantic_diversity_proxy',
                                     seconds=time.monotonic()-semantic_start,status='attempt_retained'))
                q.save_ledger()
            subprocess.run([os.sys.executable,str(ROOT/'scripts/physics_guided_analyze.py'),'--output',str(out)],check=True)
            gate=json.loads((out/'progression_gate.json').read_text())
            if not gate['pass']:
                q.state('COMPLETE_GATE_NOT_MET',reason=gate['reason'],g1='not_run',downstream='not_run')
            else:
                q.state('HUMAN_GATE_PASSED_PENDING_VALIDATED_G1_AND_TRAINING_ADAPTERS',
                        reason='No unverified robot/data adapter is automatically executed')
        except Exception as error:
            q.state('FAILED_RETAINED',error=repr(error)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    main(Path(p.parse_args().output).resolve())
