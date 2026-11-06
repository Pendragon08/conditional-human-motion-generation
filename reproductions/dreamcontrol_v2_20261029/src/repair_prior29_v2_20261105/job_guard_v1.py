"""Run one new repair job with immutable sources and project resource limits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def resources(root, pgid=None):
    project, child = {}, {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            group = os.getpgid(int(p.name))
            if str(root) not in cmd and group != pgid:
                continue
            rss = int((p / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
            project[int(p.name)] = rss
            if group == pgid:
                child[int(p.name)] = rss
        except (OSError, ValueError, ProcessLookupError):
            pass
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    used = {}
    for line in gpu.splitlines():
        pid, mb = [x.strip() for x in line.split(',')]
        if mb.isdigit():
            used[int(pid)] = used.get(int(pid), 0) + int(mb) * 1024**2
    return {'free_data_bytes': shutil.disk_usage(root).free,
            'job_rss_bytes': sum(child.values()), 'project_rss_bytes': sum(project.values()),
            'job_gpu_bytes': sum(used.get(p, 0) for p in child),
            'project_gpu_bytes': sum(used.get(p, 0) for p in project),
            'job_pids': sorted(child)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--mode', choices=['preflight', 'smoke', 'train'], required=True)
    ap.add_argument('--python', required=True)
    a = ap.parse_args()
    stage = a.stage.resolve(strict=True)
    cfg = json.loads((stage / 'config.json').read_text())
    binding = json.loads((stage / 'source_binding.json').read_text())
    for rel, sha in binding['files'].items():
        if digest(stage / rel) != sha:
            raise RuntimeError('bound source mismatch: ' + rel)
    out = stage / ('guard_' + a.mode + '_v1')
    out.mkdir(exist_ok=False)
    status = {'status': 'checking', 'mode': a.mode, 'guard_pid': os.getpid(), 'started_unix': time.time()}
    write(out / 'status.json', status)
    proc = None
    try:
        root = Path(cfg['project_root'])
        stat = resources(root)
        reserve = cfg['disk_floor_bytes']
        if a.mode == 'train':
            reserve = max(11 * 1024**3, reserve + 5 * cfg['checkpoint_size_budget'])
        if stat['free_data_bytes'] < reserve:
            raise RuntimeError('insufficient start disk: %s < %s' % (stat['free_data_bytes'], reserve))
        if stat['project_gpu_bytes'] >= 24 * 1024**3 or stat['project_rss_bytes'] >= 64 * 1024**3:
            raise RuntimeError('project resource budget unavailable')
        with (out / 'worker.log').open('xb') as log:
            proc = subprocess.Popen([a.python, '-B', '-u', str(stage / 'src/train.py'), '--config',
                                     str(stage / 'config.json'), '--mode', a.mode],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            status.update(status='running', worker_pid=proc.pid, worker_pgid=proc.pid)
            write(out / 'status.json', status)
            while proc.poll() is None:
                stat = resources(root, proc.pid)
                status.update(resources=stat, updated_unix=time.time())
                write(out / 'status.json', status)
                if stat['free_data_bytes'] < cfg['disk_floor_bytes']:
                    raise RuntimeError('runtime disk floor reached')
                if stat['job_gpu_bytes'] > 12 * 1024**3 or stat['job_rss_bytes'] > 32 * 1024**3:
                    raise RuntimeError('job memory limit exceeded')
                if stat['project_gpu_bytes'] > 24 * 1024**3 or stat['project_rss_bytes'] > 64 * 1024**3:
                    raise RuntimeError('project memory limit exceeded')
                time.sleep(5)
            if proc.returncode:
                raise RuntimeError('worker exit %d; see preserved worker.log' % proc.returncode)
        status.update(status='completed', completed_unix=time.time(), returncode=0)
        write(out / 'status.json', status)
    except BaseException as exc:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        status.update(status='failed_preserved', error=str(exc), traceback=traceback.format_exc(),
                      completed_unix=time.time(), returncode=proc.returncode if proc else None)
        write(out / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
