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
base = Path(__file__).resolve().parents[1]
verified_sources = {}
for relative, expected in recovery['source_sha256'].items():
    actual = hashlib.sha256((base/relative).read_bytes()).hexdigest()
    if actual != expected: raise ValueError('Recovery source changed: '+relative)
    verified_sources[relative] = actual
tree = {str(p.relative_to(base/'upstream')):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((base/'upstream').rglob('*.py'))}
tree_digest = hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(',',':')).encode()).hexdigest()
if tree_digest != recovery['upstream_python_tree_sha256']:
    raise ValueError('Original upstream Python source tree changed')
from inspect_checkpoint import inspect
inspection = inspect(recovery['original_manifest'])
if inspection['optimizer_updates'] != recovery['initial_optimizer_updates']:
    raise ValueError('Deserialized optimizer counter disagrees with recovery plan')
for process in Path('/proc').iterdir():
    if not process.name.isdecimal(): continue
    try:
        argv = (process/'cmdline').read_bytes().decode().split('\x00')
        if '--output' in argv and argv[argv.index('--output')+1] == recovery['original_output']:
            if any(x.endswith(('/train_full.py','/train_repaired.py')) for x in argv):
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
(arguments.output/'source_pins_verified.json').write_text(json.dumps({'sources':verified_sources,
    'upstream_python_tree_sha256':tree_digest,'upstream_python_file_count':len(tree),
    'configuration_sha256':record['configuration_sha256'],'checkpoint_inspection':inspection},indent=2)+'\n')
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled')
with (arguments.output/'supervisor_stdout.txt').open('x') as out, (arguments.output/'supervisor_stderr.txt').open('x') as err:
    result = subprocess.run(command, env=environment, stdout=out, stderr=err)
(arguments.output/'supervisor_exit.json').write_text(json.dumps({'exit_code':result.returncode,
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
for rank in [0,1]:
    health = arguments.output/f'health_rank{rank}.json'
    status = arguments.output/f'status_rank{rank}.json'
    if health.exists() and status.exists():
        row = json.loads(health.read_text()); terminal = json.loads(status.read_text())
        row.update(last_attempt_status=row['status'], process_status=terminal['status'],
                   supervisor_exit_code=result.returncode, process_running=False)
        if result.returncode == 0: row['status'] = terminal['status']
        health.write_text(json.dumps(row,indent=2)+'\n')
sys.exit(result.returncode)
