"""Run the original native SuperMimic and four object evaluations on repaired references."""

import ast
from collections import Counter
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from job_guard_v1 import digest, resources, write

STAGE = HERE.parent.parent
ROOT = STAGE.parent
LEGACY = HERE.parent / 'supermimic_legacy_v1'
OLD_NATIVE = ROOT / 'stage3_supermimic29_20261103/native_universal_20261104'
RUN = STAGE / 'supermimic_repaired_v1'
LIBRARY = RUN / 'library'
GENERATION = STAGE / 'new_references_v2'
QUALITY = STAGE / 'evaluation_repaired_v1'
TRAINING = RUN / 'main_resume_v1'
GIB, MIB = 1024**3, 1024**2
FLOOR = 8 * GIB + 356 * MIB
TASKS = ('pick', 'pour', 'bimanual_box_lift', 'box_push_pull')
COUNTS = dict.fromkeys(TASKS, 500)
DIFFUSION_PYTHON = ROOT.parent / 'reactivebfm-reproduction-20261021/env/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load(name):
    path = LEGACY / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked(path, expected):
    require(digest(path) == expected, f'Content binding changed: {path}')
    return Path(path)


def bindings():
    return read(RUN / 'source_binding.json')


def check_bindings():
    for path, sha in bindings().items():
        checked(path, sha)


def quality_inputs():
    receipt = read(GENERATION / 'generation_receipt.json')
    manifest = checked(GENERATION / 'generation_manifest.json', receipt['generation_manifest_sha256'])
    quality = read(QUALITY / 'completion_receipt.json')
    checked(QUALITY / 'report.json', quality['report_sha256'])
    cached = read(checked(QUALITY / 'cache_build_receipt.json', quality['cache_build_receipt_sha256']))
    require(receipt['status'] == 'ready_for_eval' and receipt['reference_counts'] == COUNTS,
            'Generation receipt has not accepted 500 references per task')
    require(quality['status'] == 'new_generation_three_stage_metrics_completed'
            and quality['generation_manifest_sha256'] == digest(manifest)
            and cached['generation_manifest_sha256'] == digest(manifest)
            and cached['selected_task_counts'] == COUNTS, 'Quality report does not bind the full repaired library')
    return read(manifest), digest(manifest)


def prepare_library():
    import hashlib
    import numpy as np

    generation, generation_sha = quality_inputs()
    selected = [row for row in generation['records'] if row['selected_reference']]
    require(len(selected) == 2000 and Counter(row['task'] for row in selected) == COUNTS
            and all(row['admitted'] for row in selected), 'Repaired reference count or admission mismatch')
    for field in ('id', 'initial_noise_sha256', 'feature_content_sha256', 'state_fingerprint'):
        require(len({row[field] for row in selected}) == 2000, f'Duplicate reference {field}')
    robot = OLD_NATIVE / 'source/supermimic/robot.py'
    order = ast.literal_eval(next(node.value for node in ast.parse(robot.read_text()).body
                             if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                             and t.id == 'G1_JOINT_NAMES' for t in node.targets)))
    require(len(order) == len(set(order)) == 29, 'Native joint schema changed')
    motions = LIBRARY / 'motion_csv20'
    motions.mkdir(parents=True, exist_ok=False)
    records = []
    for row in sorted(selected, key=lambda x: (TASKS.index(x['task']), x['candidate_id'])):
        require(shutil.disk_usage(ROOT).free >= FLOOR + 10 * MIB, 'Native CSV disk reserve exhausted')
        task, cid = row['task'], row['candidate_id']
        source = checked(row['source'], row['source_sha256'])
        checked(row['source_sample'], row['source_sample_sha256'])
        kinematic = checked(row['kinematic_report'], row['kinematic_report_sha256'])
        proof_path = checked(row['native_proof'], row['native_proof_sha256'])
        proof = read(proof_path)
        require(proof['eligible_for_calibrated_scene_reference'] and proof['native_checks']
                and all(v is True for v in proof['native_checks'].values()), 'Native admission proof failed')
        require(read(kinematic)['output_sha256'] == row['source_sha256'], 'Postoptimization binding mismatch')
        adapted = source.parent.parent.parent / 'native_input' / f'candidate_{cid:08d}' / 'report.json'
        checked(adapted, row['native_compatibility_report_sha256'])
        require(proof['kinematic_report_sha256'] == digest(adapted), 'Adapted native proof mismatch')
        with np.load(source, allow_pickle=False) as data:
            meta = json.loads(data['metadata_utf8'].tobytes())
            frames = row['source_frames']
            names = data['joint_names'].tolist()
            require(meta['task'] == task and meta['candidate_id'] == cid
                    and meta['scene_sha256'] == row['scene_sha256']
                    and meta['checkpoint_sha256'] == generation['checkpoint_sha256']
                    and set(names) == set(order) and len(names) == 29
                    and data['q'].shape == (frames, 29) and float(data['fps']) == 20
                    and np.array_equal(data['sample_times'], np.arange(frames) / 20)
                    and data['hand_closed'].shape == (frames, 2)
                    and np.isin(data['hand_closed'], [0, 1]).all(), 'Reference clock, joint or metadata mismatch')
            fingerprint = hashlib.sha256(b''.join(np.ascontiguousarray(data[k]).tobytes()
                                          for k in ('q', 'root_pos', 'root_quat', 'hand_closed'))).hexdigest()
            require(fingerprint == row['state_fingerprint'], 'Reference state fingerprint mismatch')
            matrix = np.c_[data['root_pos'], data['root_quat'][:, [1, 2, 3, 0]],
                           data['q'][:, [names.index(joint) for joint in order]]]
            require(matrix.shape == (frames, 36) and np.isfinite(matrix).all(), 'Invalid native reference state')
            require(np.max(np.abs(np.linalg.norm(data['root_quat'], axis=1) - 1)) < 1e-4,
                    'Reference orientation is not normalized')
            target = motions / f'{TASKS.index(task)}_{task}_{cid}.csv'
            np.savetxt(target, matrix, delimiter=',', fmt='%.9g')
            require(np.max(np.abs(np.loadtxt(target, delimiter=',') - matrix)) < 1e-7,
                    'Native CSV roundtrip mismatch')
            records.append({**row, 'csv': str(target), 'csv_sha256': digest(target), 'metadata': meta,
                            'kinematic_report': str(adapted), 'kinematic_report_sha256': digest(adapted),
                            'object_metadata': {'initial_position': data['task_object_position'].tolist(),
                                                'initial_quaternion_wxyz': data['task_object_quaternion_wxyz'].tolist()}})
    manifest = LIBRARY / 'native_motion_manifest.json'
    write(manifest, {'preview': False, 'reference_count': 2000, 'per_task': COUNTS,
                     'body_joint_order': order, 'source_fps': 20, 'records': records,
                     'generation_manifest_sha256': generation_sha,
                     'quality_receipt_sha256': digest(QUALITY / 'completion_receipt.json')})
    write(LIBRARY / 'status.json', {'status': 'final2000_inputs_native_loader_pending',
                                    'manifest_sha256': digest(manifest)})


def worker(mode, args):
    check_bindings()
    if mode == '--prepare-library':
        prepare_library()
        return
    sys.argv = [str(HERE), *args]
    if mode == '--native-train':
        module = load('native_tracker2000_worker_v1')
        module.N, module.FINAL = RUN, LIBRARY
    elif mode == '--body-eval':
        trainer = load('native_tracker2000_worker_v1')
        trainer.N, trainer.FINAL = RUN, LIBRARY
        module = load('native_tracker2000_eval_v1')
        module.N = RUN
    elif mode == '--object-eval':
        module = load('evaluate_native2000_tasks_v2_20261105')
        result = read(TRAINING / 'status.json')
        require(result['real_vector_steps'] == 72000, 'Incomplete native training budget')
        module.MODEL = checked(TRAINING / 'logs/last.pt', result['checkpoint']['sha256'])
        module.MODEL_SHA = result['checkpoint']['sha256']
        module.LIBRARY = checked(LIBRARY / 'native_motion_manifest.json', result['manifest_sha256'])
        module.LIBRARY_SHA = result['manifest_sha256']
        module.__file__ = str(HERE)
        if args[:1] == ['--native-policy']:
            module.policy_process(Path(args[1]))
            return
    else:
        raise ValueError(mode)
    module.main()


def controller():
    RUN.mkdir(exist_ok=False)
    state = {'pid': os.getpid(), 'status': 'waiting_for_reference_quality_evaluation', 'ppo_started': False}
    status_path = RUN / 'status.json'
    write(status_path, state)
    source_files = [HERE, HERE.parent / 'job_guard_v1.py', OLD_NATIVE / 'environment_spec.json',
                    ROOT / 'stage3_pick29_20261103/env_spec_v2.json', *LEGACY.glob('*.py'),
                    *(OLD_NATIVE / 'source/supermimic').glob('*.py')]
    for name in ('stage3_pick29_env.py', 'stage3_pick29_core.py'):
        source_files.append(ROOT / 'stage3_pick29_20261103/src' / name)
    for name in ('stage3_supported_ppo_env_v1.py', 'stage3_supported_ppo_core_v1.py'):
        source_files.append(ROOT / 'stage3_supported_references_20261104/src' / name)
    write(RUN / 'source_binding.json', {str(p): digest(p) for p in source_files})
    (RUN / 'source').symlink_to(OLD_NATIVE / 'source', target_is_directory=True)
    (RUN / 'environment_spec.json').symlink_to(OLD_NATIVE / 'environment_spec.json')
    spec = read(RUN / 'environment_spec.json')
    require(spec['spec_hash'] == '32a40546', 'Native environment version changed')
    native = spec['spec']
    cache = Path('/dev/shm') / f'dcv2_supermimic_repaired_v1_{os.getpid()}'
    cache.mkdir(mode=0o700, exist_ok=False)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': native['gpu_uuid'], 'PYTHONUNBUFFERED': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'MAX_JOBS': '1', 'TORCHINDUCTOR_COMPILE_THREADS': '1',
           'NVRTC_DISABLE_PCH': '1', 'CUDA_CACHE_DISABLE': '1',
           'XDG_CACHE_HOME': str(cache / 'xdg'), 'WARP_CACHE_PATH': str(cache / 'warp'),
           'TRITON_CACHE_DIR': str(cache / 'triton'), 'TORCHINDUCTOR_CACHE_DIR': str(cache / 'inductor'),
           'MPLCONFIGDIR': str(cache / 'mpl'), 'TMPDIR': str(cache), 'TMP': str(cache), 'TEMP': str(cache)}

    def update(label, **fields):
        state.update(status=label, updated_unix=time.time(), **fields)
        write(status_path, state)

    def wait_disk_gpu(minimum, label, gpu=True):
        while True:
            check_bindings()
            stats = resources(ROOT)
            update(label, resources=stats, required_free_bytes=minimum)
            ready = stats['free_data_bytes'] >= minimum and stats['project_rss_bytes'] <= 32 * GIB
            if gpu:
                free = subprocess.check_output(['nvidia-smi', '--id=' + native['gpu_uuid'],
                                               '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                              text=True, timeout=15)
                ready &= (int(free.strip()) * MIB >= 12 * GIB
                          and stats['project_gpu_bytes'] <= 12 * GIB
                          and shutil.disk_usage('/dev/shm').free >= 4 * GIB)
            if ready:
                return
            time.sleep(30)

    def run(label, command, timeout, process_env=env, cwd=None, terminal=None):
        check_bindings()
        with (RUN / (label + '.log')).open('xb') as log:
            child = subprocess.Popen(command, env=process_env, cwd=cwd or RUN / 'source',
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            started = time.monotonic()
            completed_since = None
            try:
                while child.poll() is None:
                    stats = resources(ROOT, child.pid)
                    update(label, child_pid=child.pid, resources=stats)
                    require(stats['free_data_bytes'] >= FLOOR and stats['job_gpu_bytes'] <= 12 * GIB
                            and stats['job_rss_bytes'] <= 32 * GIB and stats['project_gpu_bytes'] <= 24 * GIB
                            and stats['project_rss_bytes'] <= 64 * GIB
                            and sum(p.stat().st_size for p in cache.rglob('*') if p.is_file()) <= 2 * GIB
                            and time.monotonic() - started <= timeout, f'{label} exceeded a resource limit')
                    if terminal and terminal[0].exists() and read(terminal[0]).get('status') == terminal[1]:
                        completed_since = completed_since or time.monotonic()
                        if time.monotonic() - completed_since >= 40:
                            os.killpg(child.pid, signal.SIGTERM)
                            child.wait(timeout=15)
                            break
                    time.sleep(5)
                if terminal:
                    require(read(terminal[0])['status'] == terminal[1], f'{label} did not complete')
                else:
                    require(child.returncode == 0, f'{label} exited {child.returncode}')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()

    try:
        while not (QUALITY / 'completion_receipt.json').exists():
            for p in (GENERATION / 'status.json', STAGE / 'evaluation_controller_v1/status.json'):
                if p.exists():
                    require(read(p)['status'] != 'failed_preserved', f'Prerequisite failed: {p}')
            update('waiting_for_reference_quality_evaluation')
            time.sleep(30)
        quality_inputs()
        wait_disk_gpu(FLOOR + 350 * MIB, 'waiting_for_native_library_disk', gpu=False)
        run('prepare_library', [str(DIFFUSION_PYTHON), '-B', '-u', str(HERE), '--prepare-library'], 3600)
        load_report = LIBRARY / 'native_loader_check_v1/report.json'
        run('native_loader_check', [native['python'], '-I', '-B', '-u',
             str(LEGACY / 'stage3_supported_native_library_check.py'), '--prepared-dir', str(LIBRARY),
             '--output-dir', str(load_report.parent)], 3600)
        manifest = LIBRARY / 'native_motion_manifest.json'
        manifest_sha = digest(manifest)
        report = read(load_report)
        require(report['status'] == 'all2000_native_library_passed' and report['manifest_sha256'] == manifest_sha,
                'Actual native MotionLib validation failed')
        write(LIBRARY / 'completion_receipt.json', {'status': 'four_task2000_reference_dataset_completed',
              'reference_count': 2000, 'per_task': COUNTS, 'data_ready_for_native_body_tracker': True,
              'source_fps': 20, 'body_dof': 29, 'manifest': str(manifest), 'manifest_sha256': manifest_sha,
              'native_loader_report': str(load_report), 'native_loader_report_sha256': digest(load_report),
              'native_motion_dir': str(LIBRARY / 'motion_csv20')})
        wait_disk_gpu(FLOOR + 150 * MIB, 'waiting_for_capacity_resources')
        base = [native['python'], '-I', '-B', '-u', str(HERE)]
        capacity = RUN / 'capacity_8912_v1'
        cap = capacity / 'status.json'
        run('capacity', base + ['--native-train', '--phase', 'capacity', '--output', str(capacity)], 2400,
            terminal=(cap, 'native2000_capacity_passed'))
        old_logs = OLD_NATIVE / 'formal2000_sequence_v1/main_resume_v1/logs'
        full = math.ceil((old_logs / 'last.pt').stat().st_size * 1.15 / MIB) * MIB
        small = math.ceil((old_logs / 'ckpt_028748.pt').stat().st_size * 1.15 / MIB) * MIB
        disk = max(11 * GIB, FLOOR + 3 * full + 17 * small + 100 * MIB)
        wait_disk_gpu(disk, 'waiting_for_formal_supermimic_resources')
        write(RUN / 'all_four_reference_gate.json', {'training_allowed': True, 'manifest_sha256': manifest_sha,
              'capacity_report': str(cap), 'capacity_report_sha256': digest(cap)})
        early = RUN / 'early600_v1'
        run('early600', base + ['--native-train', '--phase', 'early', '--output', str(early)], 7200,
            terminal=(early / 'status.json', 'native2000_early_checkpoint_verified'))
        run('formal_native_training', base + ['--native-train', '--phase', 'main', '--early-output', str(early),
            '--output', str(TRAINING)], 24 * 3600, terminal=(TRAINING / 'status.json',
            'native2000_training_budget_completed_evaluation_pending'))
        wait_disk_gpu(FLOOR + 100 * MIB, 'waiting_for_native_body_eval_resources')
        tracking = RUN / 'body_tracking'
        run('body_tracking', base + ['--body-eval', '--training-output', str(TRAINING), '--output', str(tracking)],
            3600, terminal=(tracking / 'status.json', 'native2000_body_tracking_evaluation_completed'))
        physx = read(ROOT / 'stage3_pick29_20261103/env_spec_v2.json')
        sim_env = {**os.environ, **physx['env'], 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'}
        sim_env.pop('CUDA_VISIBLE_DEVICES', None)
        train = read(TRAINING / 'status.json')
        results = {}
        for task in TASKS:
            wait_disk_gpu(FLOOR + 100 * MIB, 'waiting_for_object_eval_resources')
            out = RUN / ('object_eval_' + task)
            run('object_eval_' + task, [physx['smoke_command'][0], '-u', str(HERE), '--object-eval',
                '--headless', '--device', 'cuda:0', '--task', task, '--num-envs', '100', '--output', str(out)],
                1200, process_env=sim_env, cwd=physx['cwd'])
            result = read(out / 'report.json')
            require(result['episode_count'] == len(result['episodes']) == 100
                    and result['success_count'] == sum(row['success'] for row in result['episodes'])
                    and result['checkpoint_sha256'] == train['checkpoint']['sha256']
                    and result['optimizer_updates'] == 0, 'Invalid downstream evaluation receipt')
            results[task] = {key: result[key] for key in ('success_count', 'success_ratio', 'control_error_m',
                             'control_error_episode_count', 'policy_state_sha256')}
            results[task].update(report=str(out / 'report.json'), report_sha256=digest(out / 'report.json'))
            update('object_evaluation_progress', object_evaluation=results)
        require(len({value['policy_state_sha256'] for value in results.values()}) == 1,
                'Evaluated policy states differ between tasks')
        completion = {'status': 'supermimic_repaired_training_and_evaluation_completed',
                      'manifest_sha256': manifest_sha, 'training': train,
                      'body_tracking_report': str(tracking / 'report.json'),
                      'body_tracking_sha256': digest(tracking / 'report.json'), 'object_evaluation': results}
        write(RUN / 'completion_receipt.json', completion)
        update(completion['status'], completion=completion)
    except BaseException as error:
        update('failed_preserved', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    if sys.argv[1:2] == ['--native-policy']:
        worker('--object-eval', sys.argv[1:])
    elif sys.argv[1:]:
        worker(sys.argv[1], sys.argv[2:])
    else:
        with (STAGE / '.supermimic_repaired_v1.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            controller()
