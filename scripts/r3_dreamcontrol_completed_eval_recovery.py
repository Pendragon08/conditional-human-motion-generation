"""One USD-I/O official evaluation of an audited completed task, then resume queue."""
import argparse
import fcntl
import json
import re
import shutil
import sys
import time
import r3_dreamcontrol_remaining_queue as q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=[t[0] for t in q.JOBS])
    parser.add_argument('--model-sha', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    task = args.task
    tag = f'r3_{task}_usd_eval_recovery_20261002'
    train_log = q.MON / f'{q.PREFIX}_{task}_train.log'
    failed_log = q.MON / f'{q.PREFIX}_{task}_eval.log'
    text = train_log.read_text(errors='replace')
    assert re.search(r'Learning iteration\s+1999/2000', text)
    run = re.findall(r'Exact experiment name requested from command line: (.+)', text)[-1].strip()
    model = q.TRAIN / 'logs/rsl_rl/g1' / run / 'model_1999.pt'
    assert q.digest(model) == args.model_sha
    assert 'malloc(): invalid size (unsorted)' in failed_log.read_text(errors='replace')
    task_state = q.MON / f'{q.PREFIX}_{task}.state'
    assert task_state.read_text().startswith('BLOCKED')
    assert 'eval_exit=1' in task_state.read_text()
    log = q.MON / (tag + '.log')
    assert not log.exists(), 'One adapted attempt only; audit an existing log'
    gate = json.loads((q.MON / 'r3_i1_live_training_view_gate.json').read_text())
    assert gate['user_confirmed_local_stream']
    entrypoint = q.TRAIN / 'scripts/reinforcement_learning/rsl_rl/play_eval.py'
    assert '--disable_fabric' in entrypoint.read_text()
    with (q.MON / (q.PREFIX + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f'COMPLETED_EVAL_RECOVERY_PREFLIGHT_PASS task={task} final1999 hash_verified envs1000 frames500', flush=True)
        if args.check:
            return
        backup = q.ROOT / 'artifacts/recovery' / (task + '_20261002_final1999')
        backup.mkdir(parents=True, exist_ok=False)
        for source in (model, train_log, failed_log, task_state, q.MON / (q.PREFIX + '.state')):
            shutil.copy2(source, backup / source.name)
        assert q.digest(backup / model.name) == args.model_sha
        q.write(backup / 'manifest.json', {'task': task, 'checkpoint_sha256': args.model_sha,
            'train_log_sha256': q.digest(train_log), 'failed_eval_log_sha256': q.digest(failed_log),
            'training_complete': True, 'retraining': False, 'root_cause': 'unconfirmed',
            'failed_eval': 'native malloc abort during simulation initialization'})
        state = q.MON / (tag + '.state')
        q.write(state, f'RUNNING task={task} official_eval_only render_io=USD envs=1000 frames=500')
        eid = 'r3_remaining_' + task + '_8192_eval_usd_io'
        rc = q.execute([str(q.ROOT / 'third_party/isaacsim-4.5.0/python.sh'), '-u', str(entrypoint),
            '--task=Isaac-Motion-Tracking-' + task + '-v0', '--device=cuda:1', '--num_envs=1000',
            '--path=' + str(model), '--headless', '--video', '--video_length=500',
            '--id=' + eid, '--disable_fabric'], log, q.environment(), 7200)
        matches = re.findall(r'Success rate:\s*([\d.]+)\s*%', log.read_text(errors='replace'))
        if rc or not matches:
            q.write(state, f'BLOCKED task={task} eval_exit={rc} rate_present={bool(matches)} no_retraining=true')
            raise RuntimeError('Adapted evaluation failed; no further automatic retry')
        videos = list((model.parent / 'videos/play').glob('*' + eid + '*.mp4'))
        assert len(videos) == 1 and videos[0].stat().st_size > 0
        video = videos[0]
        duration = float(q.subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=nw=1:nk=1', str(video)], text=True))
        assert duration > 0
        result = {'task': task, 'status': 'complete', 'success_percent': float(matches[-1]),
            'checkpoint': str(model), 'checkpoint_sha256': args.model_sha, 'video': str(video),
            'video_sha256': q.digest(video), 'duration_seconds': duration, 'train_exit': 0,
            'eval_exit': 0, 'num_envs': 8192, 'iterations': 2000, 'seed': 42,
            'eval_envs': 1000, 'eval_frames': 500, 'eval_environment_seed': None,
            'train_log': str(train_log), 'eval_log': str(log), 'failed_eval_logs': [str(failed_log)],
            'eval_io_adaptation': 'official --disable_fabric uses USD I/O instead of Fabric',
            'physics_policy_reward_reference_unchanged': True,
            'strict_identical_runtime_configuration': False, 'eval_attempts': 2,
            'original_native_failure_root_cause': 'unconfirmed', 'retraining': False}
        q.write(q.ROOT / f'results/r3_dreamcontrol_official_{task}_8192_eval.json', result)
        q.write(task_state, f'COMPLETE task={task} success_percent={result["success_percent"]} checkpoint={model} eval_io=USD')
        q.write(state, f'COMPLETE task={task} exit=0 video_verified USD_IO_adaptation_documented')
        print(json.dumps(result), flush=True)
    time.sleep(10)
    sys.argv = [sys.argv[0]]
    q.main()


if __name__ == '__main__':
    main()
