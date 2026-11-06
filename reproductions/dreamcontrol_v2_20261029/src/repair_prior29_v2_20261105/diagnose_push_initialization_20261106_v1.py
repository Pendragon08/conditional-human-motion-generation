"""One existing rejected push sample, changed initialization only; no sampling."""
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
OUT = S / 'push_initialization_diagnostic_v1'
PLAN = S / 'paper_pipeline_v10_launch/config.json'
PLAN_SHA = 'e074e9b85998355a6588a0d1d711495014e587b724adaa5ee70a307b3986798a'
TASK = 'box_push_pull'
CID = 20300007
COHORT = S / 'new_references_paper_v10' / TASK / 'cohort_0007'
sys.path.insert(0, str(S / 'src/paper_protocol_v10'))


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


def prepare():
    plan = bind()
    assert OUT.exists() and (OUT / 'cpu_report.json').exists()
    if (OUT / 'launch.json').exists():
        print((OUT / 'launch.json').read_text())
        return
    assert sha(OUT / 'cpu_report.json') == '2ff7b2bbb3911bd982df1a99a342533bf88bc809eed48d76353df9741824342a'
    record = read(OUT / 'cpu_report.json')
    assert record['candidate_id'] == CID and record['same_task_scene_and_phase']
    assert record['chosen']['wrist_max_m'] < .15 and record['chosen']['min_wrist_table_sdf_m'] > .03
    assert read(S / 'new_references_paper_v10/status.json')['status'] == 'failed_preserved'
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
    report['initialization_diagnostic'] = {'seed_candidate_id': record['seed_candidate_id'], 'initial_state_only': True,
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
    for _ in range(100):
        if (OUT / 'launch.json').exists(): break
        time.sleep(.1)
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
