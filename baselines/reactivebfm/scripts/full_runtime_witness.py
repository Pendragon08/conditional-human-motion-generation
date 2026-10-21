"""One documented non-mutating runtime witness: parity, DDP probe and physics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser(); p.add_argument('--runtime-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True); a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
base = a.runtime_root / 'code'; python = sys.executable; started = time.monotonic()
env = os.environ.copy(); env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', MUJOCO_GL='egl')
def run(name, arguments, devices):
    e = env.copy(); e['CUDA_VISIBLE_DEVICES'] = devices
    cmd = [python, *map(str, arguments)]
    (a.output / (name + '_command.json')).write_text(json.dumps(cmd) + '\n')
    with (a.output / (name + '_stdout.txt')).open('w') as out, (a.output / (name + '_stderr.txt')).open('w') as err:
        result = subprocess.run(cmd, env=e, stdout=out, stderr=err, timeout=240)
    if result.returncode: raise RuntimeError(name + ' failed: ' + str(result.returncode))
run('parity', [base/'scripts/check_recomputation.py', '--runtime-root', a.runtime_root, '--output', a.output/'parity.json'], '1')
run('ddp', ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2',
    base/'scripts/train_full.py', '--config', base/'configs/full_training_v1.json', '--output', a.output/'ddp', '--probe-updates', '4'], '0,1')
ranks = [json.loads((a.output/'ddp'/f'status_rank{i}.json').read_text()) for i in [0,1]]
assert all(r['status']=='PROBE_COMPLETED' and r['optimizer_updates']==4 and r['batch_size_global']==1024 for r in ranks)
run('hold', [base/'scripts/check_tracker_hold.py', '--runtime-root', a.runtime_root, '--output', a.output/'hold', '--steps', '400'], '')
run('loop', [base/'scripts/run_closed_loop.py', '--runtime-root', a.runtime_root, '--checkpoint',
    a.runtime_root/'runs/pilot_v1/checkpoints/model000000100.pt', '--output', a.output/'loop', '--steps', '100'], '1')
hold = json.loads((a.output/'hold/receipt.json').read_text()); loop = json.loads((a.output/'loop/receipt.json').read_text())
assert hold['finite'] and hold['upright_entire_rollout_height_proxy']
assert loop['status']=='PHYSICAL_CLOSED_LOOP_EXECUTED' and loop['policy_steps']==100 and loop['mujoco_integrator_steps']==400
result = {'status':'FULL_RUNTIME_WITNESS_PASSED', 'ddp_optimizer_updates_per_rank':[r['optimizer_updates'] for r in ranks],
    'global_batch':1024, 'parity':json.loads((a.output/'parity.json').read_text()), 'hold':hold,
    'closed_loop_policy_steps':loop['policy_steps'], 'closed_loop_integrator_steps':loop['mujoco_integrator_steps'],
    'closed_loop_planner_calls':len(loop['planner_calls']), 'pilot_model_stayed_upright':loop['upright_entire_rollout_height_proxy'],
    'elapsed_seconds':time.monotonic()-started,'full_training_completed':False,'paper_performance_reproduced':False}
(a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
