"""One bounded 20-update DDP recovery witness; never resumes a running job.

Uses this experiment's existing SSH key and immutable deployed v4 source.
No dependency installation, deletion or checkpoint save is performed by the probe.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-i',
       'C:/Users/Pende/.ssh/id_ed25519', 'ouyanghao@10.147.17.32']
root = '/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022'
run = root + '/probe_fp64_sdpa_v4'


def remote(source):
    process = subprocess.run(ssh + ['python3', '-'], input=source, text=True,
                             capture_output=True, encoding='utf-8', timeout=180)
    if process.returncode:
        raise RuntimeError(process.stderr)
    return json.loads(process.stdout)


launch = remote('''
from pathlib import Path
import os,json,subprocess,shutil
root=Path(ROOT)
run=root/'probe_fp64_sdpa_v4'
if run.exists():raise RuntimeError('Witness output already exists')
for p in Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  argv=(p/'cmdline').read_bytes().decode().split(chr(0))
  if any(x.endswith(('/train_full.py','/train_repaired.py')) for x in argv):
   raise RuntimeError('Existing training detected; do not launch a competing probe')
 except (OSError,UnicodeError):pass
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
if any(int(line.split(',')[1])<32000 for line in gpu.strip().splitlines()):
 raise RuntimeError('Insufficient shared-GPU headroom for witnessed full local batch')
code=root/'recovery_code_v4'
python='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
command=[python,str(code/'scripts/launch_recovery.py'),'--config',str(code/'configs/recovery_fp64_sdpa_v4.json'),'--output',str(run),'--probe-updates','20']
with (root/'probe_fp64_sdpa_v4_launcher_stdout.txt').open('x') as out,(root/'probe_fp64_sdpa_v4_launcher_stderr.txt').open('x') as err:
 p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
print(json.dumps({'pid':p.pid,'command':command,'gpu_free_mib':gpu,'disk_before':shutil.disk_usage(root)._asdict()}))
'''.replace('ROOT', repr(root)))
(args.output/'launch.json').write_text(json.dumps(launch, indent=2)+'\n', encoding='utf-8')
print(json.dumps(launch), flush=True)
deadline = time.monotonic()+1800
while time.monotonic()<deadline:
    time.sleep(20)
    state = remote('''
from pathlib import Path
import json
p=Path(RUN)
out={}
if p.exists():
 for f in p.glob('*.json'):
  try:out[f.name]=json.loads(f.read_text())
  except ValueError:pass
print(json.dumps(out))
'''.replace('RUN', repr(run)))
    (args.output/'status.json').write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    if 'supervisor_exit.json' in state:
        logs = remote('''
from pathlib import Path
import json
p=Path(RUN)
print(json.dumps({f.name:f.read_text() for f in p.glob('*') if f.suffix in ['.txt','.jsonl']}))
'''.replace('RUN', repr(run)))
        for name, content in logs.items():
            (args.output/name).write_text(content, encoding='utf-8')
        assert state['supervisor_exit.json']['exit_code']==0, 'Probe failed; evidence retained'
        for rank in [0,1]:
            status=state[f'status_rank{rank}.json']
            health=state[f'health_rank{rank}.json']
            assert status['status']=='PROBE_COMPLETED'
            assert status['optimizer_updates']==10061
            assert not status['full_training_completed']
            assert health['optimizer_updates']==10061
            assert not list(Path(args.output).glob('*.pt'))
        result={'passed':True,'initial_updates':10041,'final_updates':10061,
                'bounded_optimizer_updates':20,'full_training_completed':False,
                'long_run_stability_verified':False,'new_checkpoint_saved':False,
                'server_output':run}
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result),flush=True)
        break
else:
    raise TimeoutError('Probe still running or launcher failed; do not duplicate launch; inspect preserved logs')
