"""One actual CPU physics clip over SSH, with existing immutable code/weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
source = Path(__file__).with_name('filter_phygile_pairs_scalebfm.py')
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
       '-i', 'C:/Users/Pende/.ssh/id_ed25519', 'ouyanghao@10.147.17.32']
code = 'expected='+repr(source_sha)+'\n'+'''import pathlib,json,subprocess,os,hashlib,shutil
root=pathlib.Path('/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/paired_screen_scalebfm_v3')
source=root/'filter_phygile_pairs_scalebfm.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
health=pathlib.Path('/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/training_fp64_sdpa_v4/health_rank0.json')
before=json.loads(health.read_text());assert before['status']=='TRAINING'
assert shutil.disk_usage('/data').free>4*1024**3
r=pathlib.Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021')
out=root/'document_witness_v2';assert not out.exists()
cmd=[str(r/'env/bin/python'),'-u',str(source),'--runtime-root',str(r),'--code-root',str(r/'code'),'--phygile-code','/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/generator_clock_fix_v1/phygile','--manifest','/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022/clock_gate_v1/output/manifest_clock_consistent.json','--output',str(out),'--limit','1','--seeds','42']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl')
result=subprocess.run(cmd,capture_output=True,text=True,env=env,cwd=str(r/'code'),timeout=240)
after=json.loads(health.read_text())
x={'exit_code':result.returncode,'command':cmd,'source_sha256':expected,'stdout':result.stdout,'stderr':result.stderr,'training_before':before,'training_after':after,'gpu_used':False}
if (out/'summary.json').is_file():x['summary']=json.loads((out/'summary.json').read_text())
(root/'document_witness_execution.json').write_text(json.dumps(x,indent=2))
print(json.dumps(x))
'''
result = subprocess.run(ssh+['python3', '-'], input=code, capture_output=True, text=True,
                        encoding='utf-8', timeout=270)
(a.output/'ssh_stdout.txt').write_text(result.stdout, encoding='utf-8')
(a.output/'ssh_stderr.txt').write_text(result.stderr, encoding='utf-8')
if result.returncode:
    raise RuntimeError('SSH witness failed; raw output retained')
x = json.loads(result.stdout)
(a.output/'execution.json').write_text(json.dumps(x, indent=2)+'\n', encoding='utf-8')
assert x['exit_code'] == 0
assert x['summary']['status'] == 'AUXILIARY_PHYSICAL_SCREEN_COMPLETED'
assert x['summary']['actual_control_calls'] == 123
assert x['summary']['all_seed_pass_sequences'] == 1
assert x['summary']['inputs_unchanged'] and x['summary']['policy_parameters_unchanged']
assert x['training_after']['status'] == 'TRAINING'
assert x['training_after']['optimizer_updates'] >= x['training_before']['optimizer_updates']
print('PAIR_FILTER_DOCUMENT_WITNESS_PASSED', json.dumps({
    'exit_code': x['exit_code'], 'actual_control_calls': 123, 'gpu_used': False,
    'mean_mpjpe_m': x['summary']['records'][0]['mean_mpjpe_m'],
    'training_updates_before': x['training_before']['optimizer_updates'],
    'training_updates_after': x['training_after']['optimizer_updates']}))
