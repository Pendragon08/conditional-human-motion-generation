"""Bounded fresh-reader invocation; reuses the existing environment, no installs."""
import argparse,json,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 script=r'''from pathlib import Path
import os,subprocess,json,hashlib
root=Path('/data/code/ouyanghao/oyh/paper-faithful-reproduction-20261022')
runtime=Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021')
source=root/'code_v4/run_reactive_async.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='2728711d6ea5fb8f371ae5970f01adcb12c4f1196f7e9e334ec0df0368731df1'
output=root/'doc_witness_v1'
if output.exists():raise FileExistsError(output)
before=[json.loads(f.read_text()) for f in (runtime/'repair_20261022/training_fp32_v3').glob('health_rank*.json')]
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
assert int(free[1])>8192,'Insufficient free GPU1 memory for bounded inference'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',MUJOCO_GL='egl',TORCHINDUCTOR_CACHE_DIR='/home/ouyanghao/paper-faithful-compile-cache-20261022',TORCHINDUCTOR_COMPILE_THREADS='2')
cmd=[str(runtime/'env/bin/python'),'-u',str(source),'--runtime-root',str(runtime),'--code-root',str(runtime/'code'),'--checkpoint',str(runtime/'repair_20261022/training_fp32_v3/checkpoints/model000010040.pt'),'--output',str(output),'--steps','200','--prompt','walk forward','--compile-planner']
with (root/'doc_witness.stdout.txt').open('wb') as out,(root/'doc_witness.stderr.txt').open('wb') as err:
 result=subprocess.run(cmd,env=env,stdout=out,stderr=err,cwd=runtime/'code',timeout=240)
receipt=json.loads((output/'receipt.json').read_text())
after=[json.loads(f.read_text()) for f in (runtime/'repair_20261022/training_fp32_v3').glob('health_rank*.json')]
print(json.dumps({'command':cmd,'exit_code':result.returncode,'receipt':receipt,'training_before':before,'training_after':after}))
'''
 cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32','python3 -']
 r=subprocess.run(cmd,input=script,capture_output=True,text=True,encoding='utf-8',timeout=270)
 (a.output/'ssh_stdout.txt').write_text(r.stdout,encoding='utf-8');(a.output/'ssh_stderr.txt').write_text(r.stderr,encoding='utf-8')
 if r.returncode:raise RuntimeError('SSH witness failed; inspect captured output')
 obj=json.loads(r.stdout);(a.output/'result.json').write_text(json.dumps(obj,indent=2)+'\n')
 assert obj['exit_code']==0 and obj['receipt']['policy_calls']==200
 assert obj['receipt']['effective_replans']>=1 and obj['receipt']['sources_unchanged'] and obj['receipt']['checkpoint_unchanged']
 print(json.dumps({'status':'PASS','policy_calls':200,'effective_replans':obj['receipt']['effective_replans']}))
if __name__=='__main__':main()
