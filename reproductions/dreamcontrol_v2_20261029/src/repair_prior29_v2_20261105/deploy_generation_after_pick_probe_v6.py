"""Resume four task generation only from independently verified Pick data."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
R=S.parent
OLD=S/'src/generation_retry_v5'
NEW=S/'src/generation_retry_v6'
DEST=S/'generation_pipeline_v6_launch'
V5=S/'generation_pipeline_v5_launch/config.json'
PICK=S/'pick_wrist_mm_projection_v1'
BANK=R/'stage3_pick29_20261103/reference_scale500_20261104/final_library_v1/reference_bank.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return {'path':str(path),'sha256':sha(path)}


def read(path):
    return json.loads(path.read_text())


def change(text,a,b):
    assert text.count(a)==1,a
    return text.replace(a,b)


def create(path,body):
    if path.suffix=='.py':
        compile(body,str(path),'exec')
    if path.exists():
        assert path.read_text()==body,path
    else:
        with path.open('x') as f:f.write(body)


def launch(command,log,env):
    with log.open('xb') as f:
        return subprocess.Popen(command,cwd=S,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                                start_new_session=True,env=env).pid


def main():
    receipt=DEST/'launch.json'
    if receipt.exists():
        print(receipt.read_text())
        return
    assert sha(V5)=='5d2cfca7b6da4cc9f6b47fb36c63db05ddb6cf8f26244a8189f2486b7a81403a'
    assert not (S/'new_references_v6').exists()
    assert read(S/'new_references_v5/status.json')['status']=='failed_preserved'
    for pid in (1864015,1864016,1864017,1864018):
        cmd=Path(f'/proc/{pid}/cmdline')
        assert not cmd.exists() or str(S).encode() not in cmd.read_bytes()
    audit=PICK/'independent_source_readback_v1.json'
    assert sha(audit)=='09b04f5a60bdf190b352db60ce1dc2b0778d4ac434c3508bd02253f965638404'
    seed=PICK/'result.json'
    assert sha(seed)==read(audit)['source_result_sha256']=='7a8541c64e139d749bd795920892bd63da7e024384581d2388119a34070a3a40'
    row=read(seed)['records'][0]
    assert row['candidate_id']==10000007 and row['admitted'] and row['source_sha256']==read(audit)['output_sha256']
    assert sha(BANK)=='01767322fe964bd2524823d7d922ca91ec3931b73ad346a6f726d571b5f59572'
    plan=read(V5)
    NEW.mkdir(exist_ok=True)
    for spec in plan['sources'].values():
        previous=Path(spec['path'])
        if previous.parent!=OLD:continue
        assert sha(previous)==spec['sha256']
        body=previous.read_text()
        if previous.name=='pipeline_controller_v2.py':
            body=change(body,'    solver_module = import_file(bound(plan["sources"]["postopt"]), "_repaired_postopt_bound")\n','')
            body=change(body,'    frames = records[0]["source_frames"]',
                '''    kind = "postopt_pick" if records[0]["task"] == "pick" else "postopt"
    solver_module = import_file(bound(plan["sources"][kind]), "_repaired_postopt_bound")
    frames = records[0]["source_frames"]''')
            body=change(body,'def run_pilots(add, checkpoint_sha, output):',
                         'def run_pilots(add, checkpoint_sha, output, prior_pick):')
            body=change(body,'    save(receipt_path, receipt, exclusive=True)\n    for task, details in tasks.items():',
                '''    tasks["pick"] = {"passed": True, "attempts": prior_pick}
    save(receipt_path, receipt, exclusive=True)
    for task, details in tasks.items():
        if details["passed"]:
            continue''')
            body=change(body,'    largest_bytes_per_candidate = 0',
                '''    largest_bytes_per_candidate = 0
    prior_pick = []
    audit = read(bound(plan["pick_seed_audit"]))
    for offset, binding in enumerate(plan["resume_pick_cohorts"]):
        receipt = bound(binding)
        result = read(receipt)
        row = result["records"][0]
        require(len(result["records"]) == 1 and row["task"] == "pick"
                and row["candidate_id"] == 10000000+offset
                and row["checkpoint_sha256"] == checkpoint_sha, "invalid resumed Pick source")
        verify_hash(row["source_sample"], row["source_sample_sha256"])
        verify_hash(row["kinematic_report"], row["kinematic_report_sha256"])
        if row["admitted"]:
            require(offset==7 and sha(receipt)==audit["source_result_sha256"]
                    and row["source_sha256"]==audit["output_sha256"]
                    and row["native_proof_sha256"]==audit["native_proof_sha256"], "resumed Pick audit mismatch")
            verify_hash(row["source"], row["source_sha256"])
            verify_hash(row["native_proof"], row["native_proof_sha256"])
            proof = read(Path(row["native_proof"]))
            require(proof["eligible_for_calibrated_scene_reference"] and all(proof["native_checks"].values()),
                    "resumed Pick native proof did not pass")
        rows.append(row)
        prior_pick.append({"candidate_id":row["candidate_id"], "passed":row["admitted"],
                           "initial_noise_seed":row["initial_noise_seed"], "reused_receipt_sha256":sha(receipt)})
        cohorts.append({"path":str(receipt), "sha256":sha(receipt), "task":"pick",
                        "generated":1, "kinematic_pass":int(row["admitted"]), "native_pass":int(row["admitted"]),
                        "allocated_bytes":0, "reused_existing_bytes":True})
    require(len(rows)==8 and reference_counts(rows)=={"pick":1}, "resumed Pick count must be 1 from 8")
    assert_no_duplicates(rows)
    offsets["pick"] = numbers["pick"] = len(rows)''')
            body=change(body,'    run_pilots(add, checkpoint_sha, output)',
                         '    run_pilots(add, checkpoint_sha, output, prior_pick)')
        target=NEW/previous.name
        create(target,body)
        spec.update(binding(target))
    diagnosed=S/'pick_legacy_seed_diagnostic_v4/source/postopt_v2.py'
    diagnostic_plan=read(S/'pick_legacy_seed_diagnostic_v4/plan.json')
    assert sha(diagnosed)==diagnostic_plan['sources']['postopt']['sha256']
    body=diagnosed.read_text()
    start=body.index('            matching = [row for row in rows')
    end=body.index('            qa, ra = self.solve(target, goals, orient, mask, seed_q, seed_root,',start)
    body=body[:start]+'''            assert self.frames == 120
            clock = np.arange(self.frames)
            contact = int(m["grasp_frame"])
            assert 40 <= contact <= 50
            phase = np.interp(clock, [0,contact,contact+40,119], [0,40,80,119])
            lo = np.floor(phase).astype(int)
            hi = np.minimum(lo+1,119)
            blend = (phase-lo)[:,None,None]
            center = np.asarray(m["task_object_position"])
            nearby = sorted(rows,key=lambda row:np.linalg.norm(np.asarray(row["task_object_position"])-center))[:12]
            scored = []
            for row in nearby:
                state = Path(row["path"])
                assert sha_file(state) == row["sha256"]
                with np.load(state,allow_pickle=False) as z:
                    shifted = z["keypoints"][lo]*(1-blend) + z["keypoints"][hi]*blend
                    score = rms_vector(shifted-sample["decoded_world"])
                scored.append((score,row["sha256"],row))
            seed_error, _, row = min(scored,key=lambda item:item[:2])
            legacy_state = Path(row["path"])
            assert sha_file(legacy_state) == row["sha256"]
            assert sha_file(Path(row["postopt_report"])) == row["postopt_report_sha256"]
            with np.load(legacy_state, allow_pickle=False) as z:
                assert z["joint_names"].tolist() == self.names and float(z["fps"]) == 20.
                seed_q = np.column_stack([np.interp(phase,clock,q) for q in z["q"].T])
                pos = np.column_stack([np.interp(phase,clock,v) for v in z["root_pos"].T])
                rot = Slerp(clock,self.Rotation.from_quat(z["root_quat"][:,[1,2,3,0]]))(phase)
                seed_root = np.c_[rot.as_quat()[:,[3,0,1,2]],pos]
            seed_q = jnp.asarray(np.clip(seed_q,self.low+1e-6,self.high-1e-6), dtype=jnp.float32)
            seed_root = jnp.asarray(seed_root,dtype=jnp.float32)
            report["initialization_only"] = "phase-aligned archive state supplies an optimizer seed; full refit to raw with table collision"
            report["legacy_seed_sha256"] = row["sha256"]
            report["legacy_seed_candidate_id"] = json.loads(Path(row["kinematic_report"]).read_text())["candidate_id"]
            report["legacy_seed_raw_rmse_m"] = seed_error
'''+body[end:]
    body=change(body,'                if record["passed"]:\n                    chosen = (q, rb, points, wquat, record)',
        '''                if [key for key,value in checks.items() if not value] == ["hand_target"]:
                    assert sample["wrist_mask"][:,1].all() and not sample["wrist_mask"][:,0].any()
                    error = sample["targets_world"][:,21]-points[:,21]
                    norm = np.linalg.norm(error,axis=1)
                    radius = self.policy.wrist_position_max_m-.00005
                    scale = np.divide(np.maximum(norm-radius,0.),norm,out=np.zeros_like(norm),where=norm>0.)
                    delta = scale[:,None]*error
                    moved = float(np.linalg.norm(delta,axis=1).max())
                    if moved < .004:
                        projected = rb.copy()
                        projected[:,4:] += delta.astype(rb.dtype)
                        ref_metrics, ref_q, ref_points, ref_wquat = self.measure(sample,qb,projected,stage_a)
                        ref_checks = acceptance_checks(ref_metrics,self.policy)
                        report["stages"].append({"attempt":attempt,"refinement":"minimum norm wrist-ball projection",
                            "max_added_root_translation_m":moved,"metrics":ref_metrics,"checks":ref_checks,
                            "passed":all(ref_checks.values()),"source_sha256":record["sha256"]})
                        if all(ref_checks.values()):
                            record,q,rb,points,wquat = report["stages"][-1],ref_q,projected,ref_points,ref_wquat
                if record["passed"]:
                    chosen = (q, rb, points, wquat, record)''')
    pick=NEW/'postopt_pick.py'
    create(pick,body)
    plan['sources'].update(postopt_pick=binding(pick),legacy_pick_bank=binding(BANK))
    plan['pick_seed_audit']=binding(audit)
    plan['resume_pick_cohorts']=[binding(S/f'new_references_v5/pick/cohort_{i:04d}/result.json') for i in range(7)]+[binding(seed)]
    plan['output_dir']=str(S/'new_references_v6')
    DEST.mkdir(exist_ok=True)
    config=DEST/'config.json'
    create(config,json.dumps(plan,indent=2)+'\n')
    watchers={}
    for name in ('evaluation_controller','supermimic_repaired_controller'):
        old=S/f'src/{name}_v5.py'
        new=S/f'src/{name}_v6.py'
        body=old.read_text().replace('new_references_v5','new_references_v6').replace('evaluation_controller_v5','evaluation_controller_v6').replace('supermimic_repaired_v5','supermimic_repaired_v6')
        create(new,body)
        watchers[name]=new
    video=S/'src/report_video_sources_v3.py'
    create(video,(S/'src/report_video_sources_v2.py').read_text().replace('new_references_v5','new_references_v6').replace(
            'supermimic_repaired_v5','supermimic_repaired_v6').replace('report_video_sources_v2','report_video_sources_v3'))
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
    cpu={**env,**plan['postopt_env'],'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
    gen=launch([plan['sample_python'],'-B','-u',str(NEW/'pipeline_controller_v2.py'),'--plan',str(config),'--plan-sha256',sha(config)],
               DEST/'controller.log',env)
    evaluation=launch(['/usr/bin/python3','-B','-u',str(watchers['evaluation_controller'])],S/'evaluation_controller_v6.log',env)
    sm=launch(['/usr/bin/python3','-B','-u',str(watchers['supermimic_repaired_controller'])],S/'supermimic_repaired_v6.log',env)
    vid=launch([plan['postopt_python'],'-B','-u',str(video)],S/'report_video_sources_v3.log',cpu)
    result={'generation_pid':gen,'evaluation_pid':evaluation,'supermimic_pid':sm,'video_pid':vid,
            'plan_sha256':sha(config),'pick_admitted':1,'pick_attempts_preserved':8}
    create(receipt,json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
