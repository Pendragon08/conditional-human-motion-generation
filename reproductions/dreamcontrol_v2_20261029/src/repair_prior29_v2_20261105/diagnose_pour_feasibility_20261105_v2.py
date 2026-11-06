"""One Pour feasibility-focused repair, with the same raw motion and fixed Stage A anchor."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
OLD=S/'new_references_v6'
V6=S/'generation_pipeline_v6_launch/config.json'
V6_SHA='94dcaeb8493dc55dc0e59590ab8f41b6cfd9aaffcc79ac39bc9e3e265428af7e'
OUT=S/'pour_feasibility_refinement_v2'
BANK=S.parent/'stage3_supported_references_20261104/mixed2000_final_v1/native_motion_manifest.json'
BANK_SHA='6d35b617780968b45ef9d05c2014df17708d6b45bb7158438a13ec6c49df36cf'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def replace(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b)


def setup():
    if OUT.exists():
        print((OUT/'launch.json').read_text())
        return
    assert sha(V6)==V6_SHA and sha(BANK)==BANK_SHA
    plan=read(V6)
    previous=Path("/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105/pour_verified_seed_diagnostic_v1")
    prior_path=previous/'postopt/candidate_10100000/report.json'
    assert sha(prior_path)=='50739d3f171db8a1117ba1e76d7ea2c700420b4f1a32b66a45d51c326f34d0cf'
    prior=read(prior_path)
    source=previous/'postopt_pour.py'
    assert sha(source)=='5704cd44643d1bf0d60fa1a824e8c82f824d277a29ac02703f57f0a8cbc561f4'
    body=source.read_text()
    body=replace(body,'            return ((pose(vals, q, r).translation()[links] - target) / .03).flatten()',
        '''            diff = pose(vals, q, r).translation()[links] - target
            errors = jnp.sqrt(jnp.maximum(jnp.sum(diff**2,axis=-1),1e-12))
            frame_rmse = jnp.sqrt(jnp.mean(errors**2))
            return jnp.concatenate([(diff / .03).flatten(),
                                    jnp.maximum(errors-.145,0.)/.005,
                                    (jnp.maximum(frame_rmse-.049,0.)/.01)[None]])''')
    body=replace(body,'            position = (wrist_pose.translation() - goal) / .03',
        '''            position = wrist_pose.translation() - goal''')
    body=replace(body,'            angle = (jaxlie.SO3(orientation).inverse() @ wrist_pose.rotation()).log() / .20',
        '''            angle = (jaxlie.SO3(orientation).inverse() @ wrist_pose.rotation()).log()''')
    body=replace(body,'            return jnp.concatenate([(position * mask[:, None]).flatten(), (angle * mask[:, None]).flatten()]) * penalty',
        '''            distance = jnp.sqrt(jnp.maximum(jnp.sum(position**2,axis=-1),1e-12))
            rotation = jnp.sqrt(jnp.maximum(jnp.sum(angle**2,axis=-1),1e-12))
            return jnp.concatenate([jnp.maximum(distance-.028,0.)*mask/.004,
                                    jnp.maximum(rotation-.185,0.)*mask/.02])''')
    body=replace(body,'jnp.abs(vals[q] - vals[qp]) - .25, 0.) / .05','jnp.abs(vals[q] - vals[qp]) - .285, 0.) / .009')
    body=replace(body,'            return jnp.concatenate([keypoint, root[None], joint]) * 4.',
        '''            rms = jnp.sqrt(jnp.maximum(jnp.mean(jnp.sum(delta**2,axis=-1)),1e-12))
            return jnp.concatenate([keypoint, root[None], joint, (jnp.maximum(rms-.029,0.)/.005)[None]]) * 4.''')
    start=body.index('            # The old, immutable state supplies only an optimizer start.')
    end=body.index('            with (output / "stage_a_decoded_realization.npz").open("xb")',start)
    body=body[:start]+'''            prior_dir = Path(__file__).resolve().parent.parent / "pour_verified_seed_diagnostic_v1/postopt/candidate_10100000"
            prior_report = prior_dir / "report.json"
            assert sha_file(prior_report) == "50739d3f171db8a1117ba1e76d7ea2c700420b4f1a32b66a45d51c326f34d0cf"
            prior = json.loads(prior_report.read_text())
            assert m["candidate_id"] == 10100000 and expected_sha == prior["source_sample_sha256"]
            anchor = prior_dir / "stage_a_decoded_realization.npz"
            restart = prior_dir / "stage_b_03_raw.npz"
            assert sha_file(anchor) == prior["stage_a_sha256"] and sha_file(restart) == prior["stages"][2]["sha256"]
            with np.load(anchor,allow_pickle=False) as z:
                qa,ra = z["q"],z["root"]
            with np.load(restart,allow_pickle=False) as z:
                restart_q = np.clip(z["q"],self.low+1e-6,self.high-1e-6)
                restart_root = z["root"]
            report["initialization_only"] = "existing fitted state for one feasibility-focused repair; original Stage A anchor preserved"
            report["source_anchor_sha256"] = sha_file(anchor)
            report["source_initial_state_sha256"] = sha_file(restart)
'''+body[end:]
    body=replace(body,'enumerate((1., 2., 4.), 1)','enumerate((1.,), 1)')
    body=replace(body,'qb, rb = self.solve(target, goals, orient, mask, jnp.asarray(qa_projected), jnp.asarray(ra),',
                 'qb, rb = self.solve(target, goals, orient, mask, jnp.asarray(restart_q,dtype=jnp.float32), jnp.asarray(restart_root),')
    OUT.mkdir()
    solver=OUT/'postopt_pour.py'
    compile(body,str(solver),'exec')
    solver.write_text(body)
    plan['sources']['postopt_diagnostic']={'path':str(solver),'sha256':sha(solver)}
    plan['sources']['legacy_pour_bank']={'path':str(BANK),'sha256':BANK_SHA}
    plan_path=OUT/'plan.json'
    plan_path.write_text(json.dumps(plan,indent=2))
    env={**os.environ,**plan['postopt_env'],'PYTHONPATH':str(Path(plan['sources']['controller']['path']).parent)+os.pathsep+plan['postopt_env'].get('PYTHONPATH',''),'PYTHONDONTWRITEBYTECODE':'1'}
    with (OUT/'controller.log').open('xb') as f:
        proc=subprocess.Popen([plan['postopt_python'],'-B','-u',str(Path(__file__).resolve()),'--worker'],
            cwd=S,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'pid':proc.pid,'plan_sha256':sha(plan_path),'solver_sha256':sha(solver)}
    (OUT/'launch.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


def work():
    import pipeline_controller_v2 as p

    plan_path=OUT/'plan.json'
    plan=read(plan_path)
    cfg=p.load_config(p.bound(plan['training_config']))
    candidate=None
    try:
        p.save(OUT/'status.json',{'status':'waiting_v6_pour_gate','new_samples':0})
        deadline=time.monotonic()+3600
        while True:
            state=read(OLD/'status.json')
            pilot=read(OLD/'pilot_receipt.json')
            if pilot['pilots']['pour']['passed']:
                p.save(OUT/'status.json',{'status':'skipped_v6_pour_passed','new_samples':0})
                return
            if state['status']=='failed_preserved':
                assert pilot['status']=='pilot_quality_budget_exhausted' and pilot['failed_task']=='pour'
                assert len(pilot['pilots']['pour']['attempts'])==8
                break
            if time.monotonic()>deadline:
                raise TimeoutError('v6 Pour gate did not finish within diagnostic wait')
            time.sleep(20)
        reports=[]
        for index in range(8):
            cohort=OLD/'pour'/f'cohort_{index:04d}'
            d=cohort/'postopt'/f'candidate_{10100000+index:08d}'/'report.json'
            r=read(d)
            assert r['candidate_id']==10100000+index and r['status']=='generated_candidate_rejected'
            assert sha(Path(r['source_sample']))==r['source_sample_sha256']
            if r.get('stage_a_decoded_vector_rmse_m') is not None:
                reports.append((r['stage_a_decoded_vector_rmse_m'],r['candidate_id'],cohort,r,d))
        _,candidate,cohort,prior,prior_path=min(reports,key=lambda item:item[:2])
        assert candidate == 10100000
        out=OUT/'postopt'/f'candidate_{candidate:08d}'
        solver=p.bound(plan['sources']['postopt_diagnostic'])
        cmd=[plan['postopt_python'],'-B','-u',str(solver),'--sample',prior['source_sample'],
             '--sample-sha256',prior['source_sample_sha256'],'--output-dir',str(out),
             '--urdf',plan['urdf']['path'],'--fk-module',plan['fk']['path']]
        p.save(OUT/'chosen.json',{'candidate_id':candidate,'baseline_report_sha256':sha(prior_path),
             'selection':'lowest baseline Stage A raw fidelity RMSE among the existing eight evaluated candidates'},exclusive=True)
        try:
            p.phase(plan_path,sha(plan_path),plan,cfg,OUT,OUT,'postopt',cmd,os.environ.copy(),str(OUT))
        except ValueError:
            if not (out/'report.json').exists():
                raise
        report=read(out/'report.json')
        if not report['passed_requested_kinematic_checks']:
            p.save(OUT/'status.json',{'status':report['status'],'candidate_id':candidate,
                'report_sha256':sha(out/'report.json'),'admitted_references':0,'new_samples':0})
            return
        rows=p.producer_records(cohort/'raw','pour',[candidate],report['checkpoint_sha256'])
        staged=p.stage_native(plan,'pour',OUT/'postopt',OUT/'native_input',rows)
        assert len(staged)==1
        native=read(p.bound(plan['native_runtime']))
        script=p.bound(plan['sources']['native_supported'])
        env=p.runtime_env(native['env'],remove_visibility=True)
        env['PYTHONPATH']=str(script.parent)+os.pathsep+env.get('PYTHONPATH','')
        native_out=OUT/'native'
        argv=[native['python'],'-B','-u',str(script),'--headless','--device',plan['native_device'],
              '--output-dir',str(native_out),'--task','pour','--candidate-dir',str(staged[0][1])]
        p.phase(plan_path,sha(plan_path),plan,cfg,OUT,OUT,'native',argv,env,native['cwd'],
                (native_out/'report.json','native_task_replay_completed'))
        p.native_proofs('pour',staged,native_out)
        n=int(rows[0]['admitted'])
        p.save(OUT/'result.json',{'status':'one_pour_native_probe_complete','candidate_id':candidate,
               'new_samples':0,'admitted_references':n,'records':rows},exclusive=True)
        p.save(OUT/'status.json',{'status':'one_pour_native_probe_complete','candidate_id':candidate,
               'admitted_references':n,'result_sha256':sha(OUT/'result.json')})
    except BaseException:
        (OUT/'traceback.txt').write_text(traceback.format_exc())
        p.save(OUT/'status.json',{'status':'failed_preserved','candidate_id':candidate,'admitted_references':0})
        raise


if __name__=='__main__':
    work() if sys.argv[1:]==['--worker'] else setup()
