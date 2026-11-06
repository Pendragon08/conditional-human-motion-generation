"""One existing rejected box sample, changed initialization only; no sampling."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

S = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
OUT = S / 'box_initialization_diagnostic_v1'
PLAN = S / 'paper_pipeline_v8_launch/config.json'
PLAN_SHA = 'eb3c79702a08589d7eb73a40605e1f0d1515254420bdfbd20bfbf9de72d2a0c2'
TASK = 'bimanual_box_lift'
CID = 20200000
COHORT = S / 'new_references_paper_v8' / TASK / 'cohort_0000'
SEED_COHORT = S / 'new_references_paper_v8' / TASK / 'cohort_0004'
sys.path.insert(0, str(S / 'src/paper_protocol_v8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p)
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(value, indent=2, allow_nan=False))
    t.replace(p)


def bind():
    assert sha(PLAN) == PLAN_SHA
    plan = read(PLAN)
    for entry in plan['sources'].values():
        assert sha(entry['path']) == entry['sha256']
    for key in ('urdf', 'fk'):
        assert sha(plan[key]['path']) == plan[key]['sha256']
    return plan


def cpu_prepare(plan):
    import postopt_v2 as solver_module
    spec = importlib.util.spec_from_file_location('independent_box_fk', plan['fk']['path'])
    fk_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fk_module)
    fk = fk_module.URDFFK(plan['urdf']['path'])
    links = sorted(fk.links)
    kp = [links.index(k) for k in fk_module.PAPER_22_LINKS]
    seed_result = read(SEED_COHORT / 'result.json')
    seed = seed_result['records'][0]
    assert seed['candidate_id'] == 20200004 and seed['admitted'] and seed_result['native_pass'] == 1
    for field in ('source', 'source_sample', 'native_proof', 'kinematic_report'):
        assert sha(seed[field]) == seed[field + '_sha256']
    proof = read(seed['native_proof'])
    assert proof['eligible_for_calibrated_scene_reference'] and all(proof['native_checks'].values())
    with np.load(seed['source'], allow_pickle=False) as z:
        names = z['joint_names'].tolist()
        sq = z['q'].copy()
        sr = np.c_[z['root_quat'], z['root_pos']]
        assert float(z['fps']) == 20. and sq.shape == (180, 29)
        sm = json.loads(z['metadata_utf8'].tobytes())
    seed_pts = fk.forward(sr[:, 4:], sr[:, [1, 2, 3, 0]], sq, names, links=links)[:, kp]
    rows = []
    for index in range(5):
        cohort = S / 'new_references_paper_v8' / TASK / f'cohort_{index:04d}'
        rp = cohort / 'postopt' / f'candidate_{20200000 + index}' / 'report.json'
        report = read(rp)
        sample = solver_module.load_sample(report['source_sample'], report['source_sample_sha256'])
        assert sample['metadata']['task_plan']['phases'] == sm['task_plan']['phases']
        assert sample['metadata']['grasp_frame'] == sm['grasp_frame'] == 40
        assert sample['metadata']['scene_sha256'] == sm['scene_sha256']
        goal = sample['targets_world'][:, [20, 21]]
        def stats(points):
            wrist = points[:, [20, 21]]
            err = np.linalg.norm(wrist - goal, axis=-1)
            at = np.unravel_index(err.argmax(), err.shape)
            d = np.abs(wrist - [.45, 0., .4]) - [.3, .3, .4]
            sdf = np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0)
            return {'wrist_max_m': float(err.max()), 'left_max_m': float(err[:, 0].max()),
                    'right_max_m': float(err[:, 1].max()), 'max_frame_hand': list(map(int, at)),
                    'wrist_at_max': wrist[at].tolist(), 'target_at_max': goal[at].tolist(),
                    'min_wrist_table_sdf_m': float(sdf.min()),
                    'raw_rmse_m': solver_module.rms_vector(points - sample['decoded_world'])}
        states = []
        for path in [rp.parent / 'stage_a_decoded_realization.npz'] + [Path(x['path']) for x in report['stages']]:
            with np.load(path, allow_pickle=False) as z:
                q, root = z['q'].copy(), z['root'].copy()
            points = fk.forward(root[:, 4:], root[:, [1, 2, 3, 0]], q, names, links=links)[:, kp]
            states.append({'path': str(path), 'sha256': sha(path), **stats(points)})
        rows.append({'candidate_id': report['candidate_id'], 'report': str(rp), 'report_sha256': sha(rp),
                     'raw_path': report['source_sample'], 'raw_sha256': report['source_sample_sha256'],
                     'raw': stats(sample['decoded_world']), 'stages': states, 'seed_initialization': stats(seed_pts),
                     'kinematic_passed': report['passed_requested_kinematic_checks']})
    with (OUT / 'initial_state.npz').open('xb') as f:
        np.savez_compressed(f, q=sq, root=sr, joint_names=np.array(names))
    selected = next(r for r in rows if r['candidate_id'] == CID)
    result = {'scope': 'CPU FK readback; first rejected candidate fixed before experiment; new passed seed is initialization only',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'candidate_id': CID,
              'raw_path': selected['raw_path'], 'raw_sha256': selected['raw_sha256'],
              'seed_candidate_id': 20200004, 'seed_source': seed['source'], 'seed_source_sha256': seed['source_sha256'],
              'seed_native_proof': seed['native_proof'], 'seed_native_proof_sha256': seed['native_proof_sha256'],
              'seed_receipt_sha256': sha(SEED_COHORT / 'result.json'), 'seed_state_sha256': sha(OUT / 'initial_state.npz'),
              'same_task_scene_and_phase': True, 'new_samples': 0, 'comparisons': rows}
    save(OUT / 'cpu_report.json', result)
    return result


def prepare():
    plan = bind()
    if OUT.exists():
        print((OUT / 'launch.json').read_text() if (OUT / 'launch.json').exists() else 'Existing incomplete preparation; inspect')
        return
    OUT.mkdir()
    cpu_prepare(plan)
    env = {**os.environ, **plan['postopt_env'], 'PYTHONDONTWRITEBYTECODE': '1'}
    with (OUT / 'controller.log').open('xb') as log:
        child = subprocess.Popen([plan['postopt_python'], '-B', '-u', str(Path(__file__).resolve()), '--controller'],
                                 cwd=S, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    launch = {'pid': child.pid, 'source_sha256': sha(__file__), 'plan_sha256': PLAN_SHA,
              'cpu_report_sha256': sha(OUT / 'cpu_report.json'), 'new_samples': 0}
    save(OUT / 'launch.json', launch)
    print(json.dumps(launch))


def worker():
    from postopt_v2 import MotionPreservingSolverV2
    plan = bind()
    record = read(OUT / 'cpu_report.json')
    assert sha(record['raw_path']) == record['raw_sha256']
    assert sha(OUT / 'initial_state.npz') == record['seed_state_sha256']
    solver = MotionPreservingSolverV2(plan['urdf']['path'], 180, plan['fk']['path'])
    with np.load(OUT / 'initial_state.npz', allow_pickle=False) as z:
        assert z['joint_names'].tolist() == solver.names
        q = np.clip(z['q'], solver.low + 1e-6, solver.high - 1e-6).astype(np.float32)
        root = z['root'].astype(np.float32)
    solver.initial_state = lambda decoded: (q.copy(), root.copy())
    folder = OUT / 'postopt' / f'candidate_{CID}'
    report = solver.process(record['raw_path'], folder, record['raw_sha256'])
    report['initialization_diagnostic'] = {'seed_candidate_id': 20200004, 'initial_state_only': True,
                                         'raw_or_target_modified': False, 'losses_or_thresholds_modified': False,
                                         'cpu_report_sha256': sha(OUT / 'cpu_report.json'), 'wrapper_sha256': sha(__file__)}
    if report.get('output'):
        with np.load(report['output'], allow_pickle=False) as z:
            report['initialization_diagnostic'].update(
                output_vs_seed_max_joint_rad=float(np.max(np.abs(z['q'] - q))),
                output_vs_seed_max_root_m=float(np.linalg.norm(z['root_pos'] - root[:, 4:], axis=-1).max()))
    save(folder / 'report.json', report)
    if report['status'] == 'failed_preserved':
        raise RuntimeError('Preserved solver failure')


def controller():
    import pipeline_controller_v2 as p
    plan = bind()
    cfg = p.load_config(p.bound(plan['training_config']))
    try:
        # Reserve the observed 10.6 GB worker allocation before starting this second
        # bounded job. Both controllers continue measuring the full project total.
        deadline = time.monotonic() + 3600
        while True:
            snap = p.resource_snapshot(Path(cfg['project_root']))
            p.resource_guard(snap, plan['cohort_write_budget_bytes'])
            if snap['project_gpu_bytes'] + 11 * 1024**3 <= p.LIMITS['aggregate_gpu']:
                break
            save(OUT / 'status.json', {'status': 'waiting_for_diagnostic_resources', 'resources': snap, 'new_samples': 0})
            if time.monotonic() > deadline:
                raise TimeoutError('bounded diagnostic resource wait')
            time.sleep(15)
        launch = read(OUT / 'launch.json')
        assert sha(__file__) == launch['source_sha256'] and sha(OUT / 'cpu_report.json') == launch['cpu_report_sha256']
        p.phase(PLAN, PLAN_SHA, plan, cfg, OUT, OUT, 'postopt',
                [plan['postopt_python'], '-B', '-u', str(Path(__file__).resolve()), '--worker'],
                os.environ.copy(), str(OUT))
        report_path = OUT / 'postopt' / f'candidate_{CID}' / 'report.json'
        report = read(report_path)
        if not report['passed_requested_kinematic_checks']:
            save(OUT / 'status.json', {'status': 'initialization_diagnostic_rejected', 'report_sha256': sha(report_path), 'new_samples': 0})
            return
        rows = p.producer_records(COHORT / 'raw', TASK, [CID], report['checkpoint_sha256'])
        staged = p.stage_native(plan, TASK, OUT / 'postopt', OUT / 'native_input', rows)
        native = read(p.bound(plan['native_runtime']))
        script = p.bound(plan['sources']['native_supported'])
        env = p.runtime_env(native['env'], remove_visibility=True)
        env['PYTHONPATH'] = str(script.parent) + os.pathsep + env.get('PYTHONPATH', '')
        native_out = OUT / 'native'
        argv = [native['python'], '-B', '-u', str(script), '--headless', '--device', plan['native_device'],
                '--output-dir', str(native_out), '--task', TASK, '--candidate-dir', str(staged[0][1])]
        p.phase(PLAN, PLAN_SHA, plan, cfg, OUT, OUT, 'native', argv, env, native['cwd'],
                (native_out / 'report.json', 'native_task_replay_completed'))
        p.native_proofs(TASK, staged, native_out)
        result = {'status': 'initialization_and_native_passed' if rows[0]['admitted'] else 'native_rejected',
                  'new_samples': 0, 'candidate_id': CID, 'records': rows,
                  'report_sha256': sha(report_path), 'counted_in_library': False}
        save(OUT / 'result.json', result)
        save(OUT / 'status.json', result)
    except BaseException:
        (OUT / 'traceback.txt').write_text(traceback.format_exc())
        save(OUT / 'status.json', {'status': 'failed_preserved', 'new_samples': 0})
        raise


if __name__ == '__main__':
    {'--worker': worker, '--controller': controller}.get(sys.argv[1] if len(sys.argv) > 1 else '', prepare)()
