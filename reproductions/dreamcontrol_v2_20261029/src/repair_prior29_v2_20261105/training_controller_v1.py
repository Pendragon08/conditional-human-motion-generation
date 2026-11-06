"""Wait for the measured disk budget, then run the authorized new 10000-update job once."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from job_guard_v1 import digest, write


def main():
    stage = Path(__file__).resolve().parent.parent
    out = stage / 'training_controller_v1'
    out.mkdir(exist_ok=False)
    lock = (out / 'controller.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((stage / 'config.json').read_text())
    need = max(11 * 1024**3, cfg['disk_floor_bytes'] + 5 * cfg['checkpoint_size_budget'])
    status = {'status': 'checking_prerequisites', 'pid': os.getpid(), 'started_unix': time.time(),
              'required_start_free_bytes': need, 'formal_training_started': False,
              'source_sha256': digest(__file__), 'config_sha256': digest(stage / 'config.json')}
    write(out / 'status.json', status)
    try:
        for mode in ['preflight', 'smoke']:
            assert json.loads((stage / ('guard_' + mode + '_v1') / 'status.json').read_text())['status'] == 'completed'
        smoke = json.loads((stage / 'reports/gpu_smoke.json').read_text())
        assert smoke['real_gpu_optimizer_steps'] == 8 and smoke['checkpoint_validation']['all_fields_exact']
        assert smoke['real_adam_checkpoint_serialized_bytes'] * 1.15 < cfg['checkpoint_size_budget']
        status['gpu_smoke_sha256'] = digest(stage / 'reports/gpu_smoke.json')
        cancel = Path(cfg['project_root']) / 'stage3_supported_references_20261104/task_ppo_20261104/active_controller.json'
        assert json.loads(cancel.read_text())['status'] == 'cancelled_by_user'
        assert not (stage / 'guard_train_v1').exists()
        assert not (stage / 'runs' / cfg['run_name']).exists()
        while True:
            free = shutil.disk_usage(stage).free
            status.update(status='waiting_actual_checkpoint_disk_budget', free_data_bytes=free, updated_unix=time.time())
            write(out / 'status.json', status)
            if free >= need:
                break
            time.sleep(30)
        command = ['python3', '-B', str(stage / 'src/job_guard_v1.py'), '--stage', str(stage), '--mode', 'train',
                   '--python', '/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python']
        with (out / 'guard.log').open('xb') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            status.update(status='guard_running', guard_pid=proc.pid, guard_command=command)
            write(out / 'status.json', status)
            while proc.poll() is None:
                guard_status = stage / 'guard_train_v1/status.json'
                if guard_status.exists():
                    gs = json.loads(guard_status.read_text())
                    status.update(formal_training_started=gs['status'] in ['running', 'completed'], guard_status=gs['status'],
                                  worker_pid=gs.get('worker_pid'), updated_unix=time.time())
                    write(out / 'status.json', status)
                time.sleep(10)
        if proc.returncode:
            raise RuntimeError('new training guard exited %d; all evidence preserved' % proc.returncode)
        final = json.loads((stage / 'runs' / cfg['run_name'] / 'status.json').read_text())
        assert final['status'] == 'completed' and final['cursor']['step'] == 10000
        status.update(status='training_completed_10000_evaluation_and_regeneration_pending',
                      formal_training_started=True, completed_unix=time.time(), final_training_status=final)
        write(out / 'status.json', status)
    except BaseException as exc:
        status.update(status='failed_preserved_no_automatic_retry', error=str(exc), traceback=traceback.format_exc(),
                      completed_unix=time.time())
        write(out / 'status.json', status)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
