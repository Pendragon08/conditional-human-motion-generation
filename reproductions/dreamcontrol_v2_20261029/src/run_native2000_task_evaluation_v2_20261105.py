"""Bounded read-only GPU0 evaluation, with resource guards and per-task receipts."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from evaluate_native2000_tasks_v2_20261105 import ROOT, PICK, SUP, NATIVE, SOURCE, LIBRARY, MODEL_SHA, TASKS, FLOOR, read, sha, write

sys.path[:0] = [str(PICK / 'src'), str(NATIVE.parent / 'src')]
from stage3_pick_parallel_controller import atomic, resources, stop_group

GIB = 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, choices=(4, 100), required=True)
    parser.add_argument('--tasks', choices=TASKS, nargs='+', default=TASKS)
    args = parser.parse_args()
    assert args.output.parent == NATIVE / 'formal2000_sequence_v1'
    assert len(set(args.tasks)) == len(args.tasks)
    lock = (args.output.parent / 'frozen_task_evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(exist_ok=False)
    results = {}
    state = {'status': 'preflight', 'pid': os.getpid(), 'num_envs_per_task': args.num_envs, 'reports': results}
    status_path = args.output / 'status.json'
    atomic(status_path, state)
    evaluator = Path(__file__).with_name('evaluate_native2000_tasks_v2_20261105.py')
    source_paths = [Path(__file__), evaluator, LIBRARY, PICK / 'env_spec_v2.json', NATIVE / 'environment_spec.json']
    source_paths += [SOURCE / name for name in ('kinematics.py','mdp.py','motion.py','networks.py','robot.py','rotations.py')]
    source_paths += [PICK / 'src' / name for name in ('stage3_pick29_env.py','stage3_pick29_core.py')]
    source_paths += [SUP / 'src' / name for name in ('stage3_supported_ppo_env_v1.py','stage3_supported_ppo_core_v1.py')]
    child = None
    try:
        bindings = {str(p): sha(p) for p in source_paths}
        write(args.output / 'bindings.json', bindings)
        spec = read(PICK / 'env_spec_v2.json')
        env = {**os.environ, **spec['env'], 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        env.pop('CUDA_VISIBLE_DEVICES', None)
        ppo = read(SUP / 'task_ppo_20261104/active_controller.json')
        native = read(NATIVE / 'active_native2000_controller.json')
        assert read(native['controller_status'])['status'] == 'native2000_training_and_body_tracking_evaluation_completed'
        for task in args.tasks:
            assert all(sha(path) == digest for path, digest in bindings.items())
            free = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True, timeout=15)
            assert int(free.strip()) * 1024**2 >= 12 * GIB, 'GPU0 free memory below evaluation budget'
            assert subprocess.check_output(['df','-B1','--output=avail','/data'], text=True).splitlines()[-1].strip().isdigit()
            import shutil
            assert shutil.disk_usage('/data').free >= FLOOR
            task_output = args.output / task
            command = [spec['smoke_command'][0], '-u', str(evaluator), '--headless', '--device', 'cuda:0',
                       '--task', task, '--num-envs', str(args.num_envs), '--output', str(task_output)]
            with (args.output / (task + '.log')).open('x') as log:
                child = subprocess.Popen(command, env=env, cwd=spec['cwd'], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', task=task, child_pid=child.pid)
                atomic(status_path, state)
                start = time.monotonic()
                while child.poll() is None:
                    own = resources(child.pid)
                    other = resources(ppo['pid'])
                    assert own['gpu_bytes'] <= 12 * GIB and own['rss_bytes'] <= 32 * GIB
                    assert own['gpu_bytes'] + other['gpu_bytes'] <= 24 * GIB
                    assert own['rss_bytes'] + other['rss_bytes'] <= 64 * GIB
                    assert shutil.disk_usage('/data').free >= FLOOR
                    assert time.monotonic() - start <= 900
                    time.sleep(5)
                assert child.returncode == 0, f'{task} evaluation worker exited {child.returncode}'
            report_path = task_output / 'report.json'
            if not report_path.is_file():
                raise RuntimeError(f'{task} missing report; inspect {task_output / "failure.json"} and the worker log')
            report = read(report_path)
            assert report['episode_count'] == len(report['episodes']) == args.num_envs
            assert report['success_count'] == sum(r['success'] for r in report['episodes'])
            assert report['checkpoint_sha256'] == MODEL_SHA and report['optimizer_updates'] == 0
            results[task] = {key: report[key] for key in ('episode_count','success_count','success_ratio',
                              'control_error_m','control_error_episode_count','interaction_events',
                              'episodes_with_required_hand_contact','policy_state_sha256')}
            results[task].update(report=str(report_path), sha256=sha(report_path))
            atomic(status_path, state)
            child = None
        assert len({row['policy_state_sha256'] for row in results.values()}) == 1
        state.update(status='evaluation_completed', child_pid=None, integration_only=args.num_envs < 100,
                     checkpoint_sha256=MODEL_SHA, source_binding_sha256=sha(args.output / 'bindings.json'))
        write(args.output / 'summary.json', state)
        atomic(status_path, state)
    except BaseException:
        state.update(status='failed_preserved', traceback=traceback.format_exc())
        atomic(status_path, state)
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_group(child.pid)


if __name__ == '__main__':
    main()