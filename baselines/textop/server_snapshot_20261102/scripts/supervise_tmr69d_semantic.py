"""Bound future trained-evaluator inference only; policy remains disabled now."""
import argparse
import fcntl
import json
import os
import subprocess
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from tmr69d_trained_acceptance import ROOT,PARENT,owned,read,sha
from tmr69d_semantic_inputs import (accepted_frozen_verifier,accepted_diagnostic_metadata,
    accepted_semantic_witnesses,runtime_signature,validate_semantic_policy)

p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args()
sp=owned(a.request);request=read(sp)
pp=owned(request['policy_path']);assert sha(pp)==request['policy_sha256'];s=validate_semantic_policy(read(pp))
assert os.getuid()==1013 and sha(__file__)==s['supervisor_sha256']
assert sha(owned(ROOT/'scripts/run_tmr69d_semantic.py'))==s['executor_sha256']
for name,value in s['bound_sources'].items():assert sha(owned(ROOT/name))==value
assert request['mode'] in ['witness','diagnostic']
assert request['runtime_binding_sha256']==runtime_signature(s)
assert s['witnesses_authorized'] is True if request['mode']=='witness' else s['metrics_authorized'] is True
assert request['run']=={'author':'tmr69d_semantic_author_v1','document':'tmr69d_semantic_document_v1','diagnostic':'tmr69d_semantic32_v1'}[request['role']]
lock=(ROOT/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
accepted_frozen_verifier(s['frozen_model_binding']);accepted_diagnostic_metadata(s['diagnostic_binding'])
if request['mode']=='diagnostic':accepted_semantic_witnesses(s['witness_bindings'],s)
parent=owned(PARENT);assert not (parent/request['run']).exists()
log=parent/(request['run']+'_supervision');log.mkdir(mode=0o700,exist_ok=False)
gpu=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
assert gpu>=20*1024**3


def sample(pid):
    destination=parent/request['run']
    written=sum(p.stat().st_size for p in destination.iterdir() if p.is_file()) if destination.exists() else 0
    remaining=max(0,s['output_upper_bytes']-written)
    result=sample_resources(ROOT,pid,16*1024**3,s['reserve_bytes']+s['log_allowance_bytes']+remaining)
    result['remaining_output_upper_bytes']=remaining
    return result


cmd=['taskset','-c','0-3','/usr/bin/env','CUDA_VISIBLE_DEVICES=1','PYTHONIOENCODING=utf-8',
    'PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8',
    'TOKENIZERS_PARALLELISM=false','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
    str(ROOT/'env/bin/python'),'-B',str(ROOT/'scripts/run_tmr69d_semantic.py'),'--request',str(sp)]
seconds=900 if request['mode']=='witness' else 7200
proof=run_supervised(cmd,log/'child.log',log/'resources.jsonl',log/'process_exit.json',sample,seconds,15)
rp=parent/request['run']/'receipt.json';rec=read(rp) if rp.exists() else {}
proof.update(run=request['run'],source_sha256=sha(__file__),request_sha256=sha(sp),
    receipt_sha256=sha(rp) if rp.exists() else None,resource_sha256=sha(log/'resources.jsonl'),
    process_exit_sha256=sha(log/'process_exit.json'))
with (log/'exit.json').open('x') as stream:
    json.dump(proof,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
expected='TMR69D_REAL_SEMANTIC_EMBEDDING_WITNESS_FINISHED' if request['mode']=='witness' else 'TMR69D_SEMANTIC_DIAGNOSTIC_FINISHED'
assert proof['supervision_passed'] and rec['status']==expected and rec['output_writers_closed'] is True
