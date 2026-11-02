"""Serialize and bound one actual real-data TMR diagnostic GPU process."""
import argparse
import fcntl
import json
import os
import subprocess
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from ldm100k_verified_input import ROOT, owned, read, sha

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['supervisor_sha256']
assert sha(ROOT / 'scripts/run_tmr69d_real_witness.py') == s['executor_sha256']
for name, value in s['bound_sources'].items():
    assert sha(owned(ROOT / name)) == value
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
parent = owned(ROOT / 'evaluator_tmr_20261029')
assert not (parent / s['run']).exists()
log = parent / (s['run'] + '_supervision')
log.mkdir(mode=0o700, exist_ok=False)
gpu = int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert gpu >= 20 * 1024**3


def sample(pid):
    return sample_resources(ROOT, pid, 16 * 1024**3, 2 * 1024**3 + 128 * 1024**2)


cmd = ['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=1','PYTHONIOENCODING=utf-8','PYTHONDONTWRITEBYTECODE=1',
    'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8','TOKENIZERS_PARALLELISM=false',
    'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',str(ROOT/'env/bin/python'),'-B',str(ROOT/'scripts/run_tmr69d_real_witness.py'),'--request',str(sp)]
proof = run_supervised(cmd, log/'child.log', log/'resources.jsonl', log/'process_exit.json', sample, 900, 15)
rp = parent/s['run']/'receipt.json'
r = read(rp) if rp.exists() else {}
proof.update(run=s['run'], source_sha256=sha(__file__), request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None, resource_sha256=sha(log/'resources.jsonl'), process_exit_sha256=sha(log/'process_exit.json'))
with (log/'exit.json').open('x') as f:
    json.dump(proof, f, indent=2, allow_nan=False)
    f.flush()
    os.fsync(f.fileno())
assert proof['supervision_passed'] and r['status']=='TMR69D_REAL_DATA_GPU_WITNESS_VERIFIED' and r['output_writers_closed']
assert r['actual_diagnostic_optimizer_calls']==4 and r['selected_logical_optimizer_steps']==3
assert r['actual_formal_evaluator_updates']==r['generator_or_teacher_updates']==r['persistent_fullstate_writes']==0
