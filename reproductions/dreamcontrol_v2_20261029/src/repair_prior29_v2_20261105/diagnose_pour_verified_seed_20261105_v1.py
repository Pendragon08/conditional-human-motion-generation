"""Conditional one-candidate Pour recovery using existing raw samples."""
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
OUT=S/'pour_verified_seed_diagnostic_v1'
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
    source=Path(plan['sources']['postopt_pick']['path'])
    assert sha(source)==plan['sources']['postopt_pick']['sha256']
    body=source.read_text()
    start=body.index('            legacy = Path(')
    end=body.index('            qa, ra = self.solve(target, goals, orient, mask, seed_q, seed_root,',start)
    body=body[:start]+'''            legacy = Path("/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_supported_references_20261104/mixed2000_final_v1/native_motion_manifest.json")
            assert sha_file(legacy) == "6d35b617780968b45ef9d05c2014df17708d6b45bb7158438a13ec6c49df36cf"
            rows = [r for r in json.loads(legacy.read_text())["records"] if r["task"] == "pour"]
            assert m["task"] == "pour" and self.frames == 180 and len(rows) == 500
            center = np.asarray(m["task_object_position"])
            nearby = sorted(rows,key=lambda r:np.linalg.norm(np.asarray(r["object_metadata"]["initial_position"])-center))[:12]
            scored = []
            for row in nearby:
                path = Path(row["source"])
                assert sha_file(path) == row["source_sha256"]
                assert row["metadata"]["scene_sha256"] == m["scene_sha256"]
                assert row["metadata"]["task_plan"]["phases"] == m["task_plan"]["phases"]
                with np.load(path,allow_pickle=False) as z:
                    score = rms_vector(z["keypoints"]-sample["decoded_world"])
                scored.append((score,row["source_sha256"],row))
            score, _, row = min(scored,key=lambda item:item[:2])
            path = Path(row["source"])
            assert sha_file(path) == row["source_sha256"]
            assert sha_file(Path(row["native_proof"])) == row["native_proof_sha256"]
            with np.load(path,allow_pickle=False) as z:
                assert float(z["fps"]) == 20. and z["q"].shape == (180,29)
                assert z["joint_names"].tolist() == self.names
                seed_q = jnp.asarray(np.clip(z["q"],self.low+1e-6,self.high-1e-6),dtype=jnp.float32)
                seed_root = jnp.asarray(np.c_[z["root_quat"],z["root_pos"]],dtype=jnp.float32)
            report["initialization_only"] = "old admitted Pour state used only as a start; full refit to current raw and task"
            report["legacy_seed_sha256"] = row["source_sha256"]
            report["legacy_seed_candidate_id"] = row["candidate_id"]
            report["legacy_seed_raw_rmse_m"] = score
'''+body[end:]
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
