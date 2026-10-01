"""One official USD-I/O video evaluation after Fabric/render path failure."""
import argparse
import fcntl
import json
import re
import time
import r3_dreamcontrol_remaining_queue as q

TAG = 'r3_sit_usd_video_eval_20261001'
SHA = '4302e29fe6376452a21fe93351e616fb23ec5bb83206eb015df3b5be4bea85c7'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    model = q.TRAIN / 'logs/rsl_rl/g1/2026-10-01_16-54-27-Sit-a/model_1999.pt'
    assert q.digest(model) == SHA
    assert 'BLOCKED eval_exit=1' in (q.MON / 'r3_dreamcontrol_sit_eval_recovery_20261001.state').read_text()
    diag = (q.MON / 'r3_sit_no_camera_init_diagnostic_20261001.state').read_text()
    assert diag.startswith('STOPPED') and 'policy_load_reached=True' in diag and 'native_malloc_abort=False' in diag
    entrypoint = q.TRAIN / 'scripts/reinforcement_learning/rsl_rl/play_eval.py'
    assert '--disable_fabric' in entrypoint.read_text()
    assert q.digest(q.ROOT / 'artifacts/recovery/Sit_20261001_final1999/model_1999.pt') == SHA
    log = q.MON / (TAG + '.log')
    assert not log.exists(), 'One adapted attempt only'
    with (q.MON / (q.PREFIX + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('SIT_USD_VIDEO_EVAL_PREFLIGHT_PASS same_model_envs1000_frames500 official_disable_fabric', flush=True)
        if args.check:
            return
        state = q.MON / (TAG + '.state')
        q.write(state, 'RUNNING official_eval render_io=USD envs=1000 frames=500 physics_and_policy_unchanged')
        eid = 'r3_remaining_Sit_8192_eval_usd_io'
        rc = q.execute([str(q.ROOT / 'third_party/isaacsim-4.5.0/python.sh'), '-u', str(entrypoint),
            '--task=Isaac-Motion-Tracking-Sit-v0', '--device=cuda:1', '--num_envs=1000',
            '--path=' + str(model), '--headless', '--video', '--video_length=500',
            '--id=' + eid, '--disable_fabric'], log, q.environment(), 7200)
        matches = re.findall(r'Success rate:\s*([\d.]+)\s*%', log.read_text(errors='replace'))
        if rc or not matches:
            q.write(state, f'BLOCKED eval_exit={rc} rate_present={bool(matches)} no_retraining=true')
            raise RuntimeError('USD video eval failed; preserve queue blockage')
        videos = list((model.parent / 'videos/play').glob('*' + eid + '*.mp4'))
        assert len(videos) == 1 and videos[0].stat().st_size > 0
        video = videos[0]
        duration = float(q.subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=nw=1:nk=1', str(video)], text=True))
        assert duration > 0
        result = {'task': 'Sit', 'status': 'complete', 'success_percent': float(matches[-1]),
            'checkpoint': str(model), 'checkpoint_sha256': SHA, 'video': str(video),
            'video_sha256': q.digest(video), 'duration_seconds': duration, 'train_exit': 0,
            'eval_exit': 0, 'num_envs': 8192, 'iterations': 2000, 'seed': 42,
            'eval_envs': 1000, 'eval_frames': 500, 'eval_environment_seed': None,
            'train_log': str(q.MON / f'{q.PREFIX}_Sit_train.log'), 'eval_log': str(log),
            'failed_eval_logs': [str(q.MON / f'{q.PREFIX}_Sit_eval.log'),
                str(q.MON / 'r3_dreamcontrol_sit_eval_recovery_20261001.log')],
            'eval_io_adaptation': 'official --disable_fabric uses USD I/O instead of Fabric',
            'physics_policy_reward_reference_unchanged': True,
            'strict_identical_runtime_configuration': False,
            'original_native_failure_root_cause': 'unconfirmed; camera-disabled diagnostic loads policy',
            'retraining': False}
        q.write(q.ROOT / 'results/r3_dreamcontrol_official_Sit_8192_eval.json', result)
        q.write(q.MON / f'{q.PREFIX}_Sit.state',
            f'COMPLETE task=Sit success_percent={result["success_percent"]} checkpoint={model} eval_io=USD')
        q.write(state, 'COMPLETE exit=0 video_verified official_metric USD_IO_adaptation_documented')
        print(json.dumps(result), flush=True)
    time.sleep(10)
    q.main()


if __name__ == '__main__':
    main()
