"""One audited eval-only retry; never retrain completed Sit or erase failed logs."""
import argparse
import fcntl
import json
import re
import shutil
import time
from pathlib import Path

import r3_dreamcontrol_remaining_queue as q

MODEL_SHA = '4302e29fe6376452a21fe93351e616fb23ec5bb83206eb015df3b5be4bea85c7'
RUN = '2026-10-01_16-54-27-Sit-a'
TAG = 'r3_dreamcontrol_sit_eval_recovery_20261001'


def preflight():
    model = q.TRAIN / 'logs/rsl_rl/g1' / RUN / 'model_1999.pt'
    train_log = q.MON / f'{q.PREFIX}_Sit_train.log'
    eval_log = q.MON / f'{q.PREFIX}_Sit_eval.log'
    assert q.digest(model) == MODEL_SHA, 'Final model differs'
    assert re.search(r'Learning iteration\s+1999/2000', train_log.read_text(errors='replace'))
    assert 'malloc(): invalid size (unsorted)' in eval_log.read_text(errors='replace')
    assert (q.MON / f'{q.PREFIX}_Sit.state').read_text().startswith('BLOCKED')
    assert not (q.MON / f'{TAG}.log').exists(), 'One retry only; audit existing attempt'
    gate = json.loads((q.MON / 'r3_i1_live_training_view_gate.json').read_text())
    assert gate['user_confirmed_local_stream']
    with (q.MON / (q.PREFIX + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('SIT_EVAL_RECOVERY_PREFLIGHT_PASS final1999 hash_verified retry_only unchanged_budget', flush=True)
    return model, train_log, eval_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    model, train_log, original_eval_log = preflight()
    if args.check:
        return
    with (q.MON / (q.PREFIX + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backup = q.ROOT / 'artifacts/recovery/Sit_20261001_final1999'
        backup.mkdir(parents=True, exist_ok=False)
        for source in (model, train_log, original_eval_log,
                       q.MON / f'{q.PREFIX}_Sit.state', q.MON / f'{q.PREFIX}.state'):
            shutil.copy2(source, backup / source.name)
        assert q.digest(backup / model.name) == MODEL_SHA
        q.write(backup / 'manifest.json', {'reason': 'original evaluation native malloc abort before policy execution',
            'final_model_sha256': MODEL_SHA, 'original_train_log_sha256': q.digest(train_log),
            'original_eval_log_sha256': q.digest(original_eval_log),
            'training_complete': True, 'retraining': False, 'retry_max': 1})
        retry_log = q.MON / f'{TAG}.log'
        state = q.MON / f'{TAG}.state'
        q.write(state, 'RUNNING stage=official_eval_only envs=1000 frames=500 native_failure_cause_unproven')
        eid = 'r3_remaining_Sit_8192_eval_retry1'
        rc = q.execute([str(q.ROOT / 'third_party/isaacsim-4.5.0/python.sh'), '-u',
            'scripts/reinforcement_learning/rsl_rl/play_eval.py', '--task=Isaac-Motion-Tracking-Sit-v0',
            '--device=cuda:1', '--num_envs=1000', '--path=' + str(model), '--headless', '--video',
            '--video_length=500', '--id=' + eid], retry_log, q.environment(), 7200)
        matches = re.findall(r'Success rate:\s*([\d.]+)\s*%', retry_log.read_text(errors='replace'))
        if rc or not matches:
            q.write(state, f'BLOCKED eval_exit={rc} success_rate_present={bool(matches)} no_training_restarted')
            raise RuntimeError(f'Sit one eval-only retry failed exit={rc}; queue remains blocked')
        videos = list((model.parent / 'videos/play').glob('*' + eid + '*.mp4'))
        assert len(videos) == 1 and videos[0].stat().st_size > 0
        video = videos[0]
        duration = float(q.subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=nw=1:nk=1', str(video)], text=True))
        assert duration > 0
        result = {'task': 'Sit', 'status': 'complete', 'success_percent': float(matches[-1]),
            'checkpoint': str(model), 'checkpoint_sha256': MODEL_SHA, 'video': str(video),
            'video_sha256': q.digest(video), 'duration_seconds': duration, 'train_exit': 0,
            'eval_exit': 0, 'num_envs': 8192, 'iterations': 2000, 'seed': 42,
            'eval_envs': 1000, 'eval_frames': 500, 'eval_environment_seed': None,
            'train_log': str(train_log), 'eval_log': str(retry_log),
            'failed_eval_log': str(original_eval_log), 'eval_attempts': 2,
            'first_eval_failure': 'native malloc(): invalid size (unsorted) during simulation initialization',
            'first_eval_failure_root_cause': 'unconfirmed', 'retraining': False}
        q.write(q.ROOT / 'results/r3_dreamcontrol_official_Sit_8192_eval.json', result)
        q.write(q.MON / f'{q.PREFIX}_Sit.state',
            f'COMPLETE task=Sit success_percent={result["success_percent"]} checkpoint={model} eval_retry=1')
        q.write(state, 'COMPLETE official_eval_exit=0 training_reused final1999 video_verified')
        print(json.dumps(result), flush=True)
    # Lock is released before the original scheduler reacquires it. Completed tasks are skipped.
    time.sleep(10)
    q.main()


if __name__ == '__main__':
    main()
