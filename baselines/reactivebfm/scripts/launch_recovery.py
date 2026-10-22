"""Create a fresh recovery run; no existing task or artifact is overwritten."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--probe-updates', type=int, default=0)
arguments = parser.parse_args()
configuration = json.loads(arguments.config.read_text())
recovery = configuration['recovery']
for process in Path('/proc').iterdir():
    if not process.name.isdecimal(): continue
    try:
        argv = (process/'cmdline').read_bytes().decode().split('\x00')
        if '--output' in argv and argv[argv.index('--output')+1] == recovery['original_output']:
            if any(x.endswith('/train_full.py') for x in argv):
                raise RuntimeError('Original run must be stopped before recovery')
    except (OSError, UnicodeError, IndexError): pass
arguments.output.mkdir(parents=True, exist_ok=False)
command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
           '--nnodes=1', '--nproc_per_node=2', str(Path(__file__).with_name('train_repaired.py')),
           '--config', str(arguments.config.resolve()), '--output', str(arguments.output.resolve()),
           '--resume', recovery['model']]
if arguments.probe_updates: command += ['--probe-updates', str(arguments.probe_updates)]
record = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command':command,
          'configuration_sha256':hashlib.sha256(arguments.config.read_bytes()).hexdigest(),
          'resume':recovery['model'], 'probe_updates':arguments.probe_updates,
          'scope':'Model, EMA and AdamW continuation; random state and data order are not exact replay.'}
(arguments.output/'recovery_launch.json').write_text(json.dumps(record, indent=2)+'\n')
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')
with (arguments.output/'supervisor_stdout.txt').open('x') as out, (arguments.output/'supervisor_stderr.txt').open('x') as err:
    result = subprocess.run(command, env=environment, stdout=out, stderr=err)
(arguments.output/'supervisor_exit.json').write_text(json.dumps({'exit_code':result.returncode,
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
sys.exit(result.returncode)
