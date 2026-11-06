"""Audit the passed existing Pick, then resume the authorized sequence in new directories."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

R = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S = R / 'repair_prior29_v2_20261105'
OLD = S / 'src/paper_protocol_v7'
NEW = S / 'src/paper_protocol_v8'
DEST = S / 'paper_pipeline_v8_launch'
DIAG = S / 'paper_pick_seed_diagnostic_v2'
OLD_PLAN = S / 'paper_pipeline_v7_launch/config.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return {'path': str(path), 'sha256': sha(path)}


def create(path, body):
    if not isinstance(body, str):
        body = json.dumps(body, indent=2, allow_nan=False) + '\n'
    if path.suffix == '.py':
        compile(body, str(path), 'exec')
    if path.exists():
        assert path.read_text() == body, f'Existing different content: {path}'
    else:
        with path.open('x') as f:
            f.write(body)


def change(body, old, new):
    assert body.count(old) == 1, old
    return body.replace(old, new)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def audit(plan):
    assert sha(DIAG / 'result.json') == 'd6e2668eed726c3f914ddb7e0549404b309b604a13f4ed1944db818541ccdcc2'
    result = read(DIAG / 'result.json')
    assert result['status'] == 'seed_and_native_passed'
    row = result['records'][0]
    for key, hkey in [('source_sample', 'source_sample_sha256'), ('source', 'source_sha256'),
                      ('kinematic_report', 'kinematic_report_sha256'), ('native_proof', 'native_proof_sha256')]:
        assert sha(row[key]) == row[hkey], key
    assert row['candidate_id'] == 20000004 and row['admitted'] and not row['selected_reference']
    proof = read(row['native_proof'])
    assert all(proof['native_checks'].values()) and proof['native_replay_frames'] == 120
    assert proof['kinematic_report_sha256'] == row['native_compatibility_report_sha256']
    assert sha(DIAG / 'native_input/candidate_20000004/report.json') == proof['kinematic_report_sha256']
    report = read(row['kinematic_report'])
    assert all(report['checks'].values()) and report['output_sha256'] == row['source_sha256']
    sys.path.insert(0, str(OLD))
    from postopt_v2 import load_sample
    raw = load_sample(row['source_sample'], row['source_sample_sha256'])
    fkmod = import_file('independent_seed_resume_fk', plan['fk']['path'])
    fk = fkmod.URDFFK(plan['urdf']['path'])
    with np.load(row['source'], allow_pickle=False) as z, np.load(DIAG / 'initial_state.npz', allow_pickle=False) as seed:
        q, pos, quat, names = z['q'].copy(), z['root_pos'].copy(), z['root_quat'].copy(), z['joint_names'].tolist()
        assert names == seed['joint_names'].tolist()
        points = fk.forward(pos, quat[:, [1, 2, 3, 0]], q, names, links=fkmod.PAPER_22_LINKS)
        saved_points_error = float(np.max(np.abs(points - z['keypoints'])))
        rmse = float(np.sqrt(np.mean(np.sum((points - raw['decoded_world'])**2, axis=-1))))
        wrist = float(np.linalg.norm(points[:, 21] - raw['targets_world'][:, 21], axis=-1).max())
        dq = float(np.abs(q - seed['q']).max())
        dp = float(np.linalg.norm(pos - seed['root'][:, 4:], axis=-1).max())
    assert saved_points_error < 1e-5 and abs(rmse - report['metrics']['decoded_point_vector_rmse_m']) < 1e-7
    assert abs(wrist - report['metrics']['wrist_max_l2_m']) < 1e-7 and wrist < .15
    assert dq > 1e-6 and dp > 1e-6, 'Output must not be an unchanged legacy seed'
    evidence = {'status': 'independent_source_fk_and_native_binding_verified',
        'source_result_sha256': sha(DIAG / 'result.json'), 'output_sha256': row['source_sha256'],
        'raw_sha256': row['source_sample_sha256'], 'native_proof_sha256': row['native_proof_sha256'],
        'independent_raw_point_rmse_m': rmse, 'independent_wrist_target_max_m': wrist,
        'saved_points_max_difference_m': saved_points_error, 'output_vs_aligned_seed_max_joint_rad': dq,
        'output_vs_aligned_seed_max_root_m': dp, 'new_samples': 0,
        'scope': 'same newly sampled candidate refit, not a copied legacy reference; no physical task-success claim'}
    create(DEST / 'independent_pick_readback.json', evidence)
    return row


def main():
    if (DEST / 'launch.json').exists():
        print((DEST / 'launch.json').read_text())
        return
    assert sha(OLD_PLAN) == 'a36fcedbfccbaeb8cd065457f72e76c601d233c6bd59759db32481c7fb68db74'
    plan = read(OLD_PLAN)
    for spec in plan['sources'].values():
        assert sha(spec['path']) == spec['sha256'], spec['path']
    for pid in (1899660, 1899674, 1899675, 1899676, 1914254, 1914522):
        cmd = Path(f'/proc/{pid}/cmdline')
        assert not cmd.exists() or str(S).encode() not in cmd.read_bytes(), f'Previous process still active: {pid}'
    assert read(S / 'new_references_paper_v7/status.json')['status'] == 'failed_preserved'
    assert not (S / 'new_references_paper_v8').exists()
    DEST.mkdir(exist_ok=True)
    audit(plan)
    NEW.mkdir(exist_ok=True)
    for source in OLD.glob('*.py'):
        body = source.read_text()
        if source.name == 'pipeline_controller_v2.py':
            body = change(body, '        report = solver.process(row["source_sample"], directory, row["source_sample_sha256"])',
                '        from verified_initialization_v8 import process\n        report = process(plan, row, solver, solver_module, directory)')
            body = change(body, '    save(receipt_path, receipt, exclusive=True)\n    for task, details in tasks.items():',
                '    tasks["pick"] = {"passed": True, "attempts": prior_pick}\n    save(receipt_path, receipt, exclusive=True)\n    for task, details in tasks.items():')
            body = change(body, '    prior_pick = []\n    def add(task, count):', '''    prior_pick = []
    audit = read(bound(plan["pick_seed_audit"]))
    replacement = read(bound(plan["resumed_pick_result"]))["records"][0]
    require(sha(bound(plan["resumed_pick_result"])) == audit["source_result_sha256"], "Pick audit result mismatch")
    for offset, spec in enumerate(plan["resume_pick_cohorts"]):
        receipt = bound(spec)
        original = read(receipt)
        row = original["records"][0]
        require(len(original["records"]) == 1 and row["task"] == "pick" and row["candidate_id"] == 20000000+offset,
                "unexpected resumed Pick identity")
        verify_hash(row["source_sample"], row["source_sample_sha256"])
        verify_hash(row["kinematic_report"], row["kinematic_report_sha256"])
        require(not row["admitted"] and row["checkpoint_sha256"] == checkpoint_sha, "unexpected prior admission")
        history = {"original_cohort": str(receipt), "original_cohort_sha256": sha(receipt)}
        if row["candidate_id"] == 20000004:
            require(row["source_sample_sha256"] == replacement["source_sample_sha256"], "replacement raw changed")
            row = dict(replacement)
            for key,hkey in [("source", "source_sha256"), ("kinematic_report", "kinematic_report_sha256"),
                              ("native_proof", "native_proof_sha256")]:
                verify_hash(row[key], row[hkey])
            require(row["source_sha256"] == audit["output_sha256"] and row["native_proof_sha256"] == audit["native_proof_sha256"],
                    "replacement proof mismatch")
            proof = read(Path(row["native_proof"]))
            require(proof["eligible_for_calibrated_scene_reference"] and all(proof["native_checks"].values()), "Pick native rejected")
            history.update(refit_receipt=plan["resumed_pick_result"], audit=plan["pick_seed_audit"])
        row["resumed_provenance"] = history
        rows.append(row)
        prior_pick.append({"candidate_id":row["candidate_id"], "passed":row["admitted"],
                           "initial_noise_seed":row["initial_noise_seed"], **history})
        cohorts.append({"path":str(receipt),"sha256":sha(receipt),"task":"pick","generated":1,
                        "kinematic_pass":int(row["admitted"]),"native_pass":int(row["admitted"]),
                        "allocated_bytes":0,"reused_existing_bytes":True, **history})
    require(len(rows)==8 and reference_counts(rows)=={"pick":1}, "resume must contain one admitted from eight raw candidates")
    assert_no_duplicates(rows)
    offsets["pick"] = numbers["pick"] = 8
    save(output / "resume_receipt.json", {"status":"existing_prefix_verified", "candidate_counts":dict(offsets),
         "verified_admitted_counts":reference_counts(rows), "audit":plan["pick_seed_audit"], "new_samples":0}, exclusive=True)
    def add(task, count):''')
        create(NEW / source.name, body)
    helper = S / 'src/verified_initialization_v8.py'
    create(NEW / helper.name, helper.read_text())
    for spec in plan['sources'].values():
        source = Path(spec['path'])
        if source.parent == OLD:
            spec.update(binding(NEW / source.name))
    plan['sources']['verified_initialization'] = binding(NEW / helper.name)
    plan['pick_seed_audit'] = binding(DEST / 'independent_pick_readback.json')
    plan['resumed_pick_result'] = binding(DIAG / 'result.json')
    plan['resume_pick_cohorts'] = [binding(S / f'new_references_paper_v7/pick/cohort_{i:04d}/result.json') for i in range(8)]
    pour = S / 'paper_protocol_validation_v7/result.json'
    assert sha(pour) == '305347a6cc4b4b275a44b2541be6e74606893396f7fec058239f928a73fed318'
    plan['pour_initialization_receipt'] = binding(pour)
    plan['output_dir'] = str(S / 'new_references_paper_v8')
    plan['initialization_scope'] = 'Pick nearest validated pose, Pour validated pose, phase aligned as initial states only; full raw refit, unchanged v7 conditions/objectives/filters. Other tasks keep v7 initial state.'
    create(DEST / 'config.json', plan)
    # Reproduce the selected Pick initial state on CPU before any new generation.
    sys.path.insert(0, str(NEW))
    helpermod = import_file('verified_initialization_v8', NEW / helper.name)
    from postopt_v2 import load_sample
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace
    fkmod = import_file('seed_witness_fk', plan['fk']['path'])
    with np.load(DIAG / 'initial_state.npz', allow_pickle=False) as z:
        names = z['joint_names'].tolist()
        expected_q, expected_root = z['q'].copy(), z['root'].copy()
    limits = {x.get('name'): x.find('limit') for x in ET.parse(plan['urdf']['path']).getroot().findall('joint') if x.get('type')=='revolute'}
    fk = fkmod.URDFFK(plan['urdf']['path'])
    all_links = sorted(fk.links)
    solver = SimpleNamespace(fk=fk, names=names, all_links=all_links, kp=[all_links.index(x) for x in fkmod.PAPER_22_LINKS],
        low=np.array([float(limits[n].get('lower')) for n in names]), high=np.array([float(limits[n].get('upper')) for n in names]))
    row = read(DIAG / 'result.json')['records'][0]
    q, root, provenance = helpermod.prepare(plan, row, solver, load_sample(row['source_sample'], row['source_sample_sha256']))
    assert provenance['seed_sha256'] == read(DIAG / 'cpu_report.json')['selected']['sha256']
    assert np.max(np.abs(q-expected_q)) < 2e-6 and np.max(np.abs(root-expected_root)) < 2e-6
    prow = read(pour)['records'][0]
    pq, pr, pinfo = helpermod.prepare(plan, prow, solver, load_sample(prow['source_sample'], prow['source_sample_sha256']))
    with np.load(prow['source'], allow_pickle=False) as z:
        assert np.max(np.abs(pq-z['q'])) < 2e-6 and np.max(np.abs(pr[:,4:]-z['root_pos'])) < 2e-6
    create(DEST / 'initializer_cpu_witness.json', {'status':'passed','pick':provenance,'pour':pinfo,
        'exact_pick_diagnostic_initial_state_reproduced':True,'pour_same_phase_is_identity':True,'no_gpu':True})
    watchers = {}
    changes = ['new_references_paper','evaluation_controller_paper','evaluation_paper','supermimic_paper','report_video_sources_paper']
    for oldname, newname, key in [('evaluation_controller_v7.py','evaluation_controller_paper_v8.py','evaluation'),
                                  ('supermimic_repaired_controller_v7.py','supermimic_repaired_controller_paper_v8.py','supermimic'),
                                  ('report_video_sources_v4.py','report_video_sources_paper_v8.py','videos')]:
        body = (S / 'src' / oldname).read_text()
        for prefix in changes:
            body = body.replace(prefix+'_v7',prefix+'_v8')
        target = S / 'src' / newname
        create(target, body)
        watchers[key] = target
    create(DEST / 'source_binding.json', {str(p):sha(p) for p in [*NEW.glob('*.py'),*watchers.values(),Path(__file__)]})
    # Import/setup and resource accounting are CPU-only; no smoke retraining.
    controller = import_file('v8_plan_cpu_check', NEW / 'pipeline_controller_v2.py')
    controller.setup_plan(DEST / 'config.json')
    snapshot = controller.resource_snapshot(R)
    controller.resource_guard(snapshot, plan['cohort_write_budget_bytes'])
    create(DEST / 'prelaunch.json', {'status':'ready','plan_sha256':sha(DEST/'config.json'),
        'resources':snapshot,'new_reference_counts':{'pick':1,'pour':0,'bimanual_box_lift':0,'box_push_pull':0}})
    if '--launch' not in sys.argv:
        print((DEST / 'prelaunch.json').read_text())
        return
    for path, expected in read(DEST / 'source_binding.json').items():
        assert sha(path) == expected
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    commands = {'generation':[plan['sample_python'],'-B','-u',str(NEW/'pipeline_controller_v2.py'),
                            '--plan',str(DEST/'config.json'),'--plan-sha256',sha(DEST/'config.json')]}
    for key, path in watchers.items():
        commands[key] = [plan['sample_python'],'-B','-u',str(path)]
    processes = {}
    for key, command in commands.items():
        penv = env.copy()
        if key == 'videos':
            penv.update(plan['postopt_env']);penv['JAX_PLATFORMS']='cpu';penv['CUDA_VISIBLE_DEVICES']=''
        with (DEST / (key+'.log')).open('xb') as log:
            child = subprocess.Popen(command,cwd=S,env=penv,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes[key] = {'pid':child.pid,'argv':command}
        create(DEST / (key+'_launch.json'),processes[key])
    create(DEST / 'launch.json',{'status':'launched','plan_sha256':sha(DEST/'config.json'),'processes':processes})
    print((DEST / 'launch.json').read_text())


if __name__ == '__main__':
    main()
