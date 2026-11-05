"""After a successful GPU smoke, safely switch this project's CPU job to GPU.

The user explicitly approved sharing GPU 1. Only the specifically identified
project CPU solver is stopped; all other tasks and original artifacts remain.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SMOKE = ROOT / 'mosh_gpu_single_window_v3_20261031'
OLD = ROOT / 'mosh_000798_pyroki_fullrate_20261031'
HUMAN = ROOT / 'mosh_000798_fullrate_20261031'
NEW = ROOT / 'mosh_000798_gpu_resume_20261031'
URDF = Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
PY = sys.executable
CPU_PID = 866884
SMOKE_PID = 875576


def own_process(pid, expected):
    try:
        proc = Path('/proc') / str(pid)
        return expected in (proc / 'cmdline').read_bytes() and Path(os.readlink(proc / 'cwd')) == expected_cwd(pid)
    except (FileNotFoundError, ProcessLookupError):
        return False


def expected_cwd(pid):
    # The smoke is invoked via an absolute path; its inherited working directory
    # varies by SSH session. It is identified by its full script path instead.
    return OLD if pid == CPU_PID else Path(os.readlink(Path('/proc') / str(pid) / 'cwd'))


def main():
    out = SMOKE / 'switch_controller.log'
    with out.open('x', encoding='utf8', buffering=1) as log:
        def emit(message):
            print(message, file=log, flush=True)
        try:
            deadline = time.monotonic() + 360
            while own_process(SMOKE_PID, str(SMOKE / 'run.py').encode()):
                if time.monotonic() >= deadline: raise TimeoutError('GPU smoke did not finish')
                time.sleep(3)
            if not (SMOKE / 'report.json').is_file(): raise RuntimeError('GPU smoke failed; CPU stays running')
            compare = SMOKE / 'cpu_gpu_comparison.json'
            subprocess.run([PY, str(SMOKE / 'compare_gpu_cpu_window.py'), '--cpu-run', str(OLD),
                            '--gpu-run', str(SMOKE), '--output', str(compare)], check=True, stdout=log, stderr=log)
            if not json.loads(compare.read_text())['within_comparison_policy']:
                raise RuntimeError('GPU/CPU comparison exceeds engineering limits; CPU stays running')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(ROOT / 'jax_cuda12_plugin_0_6_2_20261031') + ':' + str(ROOT / 'pyroki_cpu_deps_20261030/site-packages')
            preflight = SMOKE / 'switch_preflight.json'
            subprocess.run([PY, str(SMOKE / 'gpu_retarget_preflight.py'), '--gpu-index', '1', '--minimum-free-mib',
                            '10240', '--allow-shared', '--output', str(preflight)], env=env, check=True, stdout=log, stderr=log)
            if not json.loads(preflight.read_text())['gpu_retarget_may_start']:
                raise RuntimeError('GPU resources unavailable; CPU stays running')
            status = (OLD / 'queue_status.txt').read_text().splitlines()
            if not status or status[-1] != 'started' or not own_process(CPU_PID, b'run.py'):
                raise RuntimeError('original CPU task is not the expected live process; no signal sent')
            os.kill(CPU_PID, signal.SIGTERM)
            emit('Sent SIGTERM only to the verified project CPU solver; waiting for queue to close.')
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                lines = (OLD / 'queue_status.txt').read_text().splitlines()
                if lines and lines[-1] == 'solver_failed' and not own_process(CPU_PID, b'run.py'): break
                time.sleep(1)
            else: raise TimeoutError('CPU queue did not confirm termination; no GPU continuation launched')
            subprocess.run([PY, str(HUMAN / 'prepare_gpu_resume.py'), '--original-run', str(OLD),
                            '--source', str(HUMAN / 'output/human_120hz.npz'), '--urdf', str(URDF),
                            '--prefix-auditor', str(OLD / 'audit_partial_pyroki_windows_v2.py'), '--output', str(NEW)],
                           check=True, stdout=log, stderr=log)
            with (OLD / 'audit.py').open('rb') as src, (NEW / 'audit.py').open('xb') as dst:
                shutil.copyfileobj(src, dst)
            launcher = '''#!/usr/bin/env bash
set -uC
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda,cpu XLA_FLAGS=--xla_gpu_autotune_level=0
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu
export PYTHONPATH=/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/jax_cuda12_plugin_0_6_2_20261031:/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/pyroki_cpu_deps_20261030/extra-packages:/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/pyroki_cpu_deps_20261030/site-packages:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/pyroki/src:/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/jaxls/src:/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/isaac_utils
printf '%s\\n' started > queue_status.txt
if nice -n 15 timeout 3600 taskset -c 18 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python run.py > solver.log 2>&1; then
  if env CUDA_VISIBLE_DEVICES= nice -n 15 /data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python audit.py > audit.log 2>&1; then
    printf '%s\\n' audited >> queue_status.txt
  else
    printf '%s\\n' audit_failed >> queue_status.txt
  fi
else
  printf '%s\\n' solver_failed >> queue_status.txt
fi
'''
            with (NEW / 'queue_gpu.sh').open('x', encoding='utf8', newline='\n') as f: f.write(launcher)
            with (NEW / 'queue.log').open('xb') as queue_log:
                proc = subprocess.Popen(['bash', str(NEW / 'queue_gpu.sh')], stdout=queue_log, stderr=queue_log,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
            emit('GPU continuation queue launched with PID %d; full results remain unaccepted.' % proc.pid)
        except Exception as exc:
            emit('SWITCH_FAILED: ' + repr(exc))
            raise


if __name__ == '__main__': main()
