from session import *
import base64
source=BASE/'scripts/run_reactive_async.py'
payload=base64.b64encode(source.read_bytes()).decode()
script=f'''from pathlib import Path
import base64,subprocess,os,json,shutil
root=Path({REMOTE!r});root.mkdir(parents=True,exist_ok=True)
code=root/'code_v4';code.mkdir(exist_ok=True)
p=code/'run_reactive_async.py'
data=base64.b64decode({payload!r})
if p.exists() and p.read_bytes()!=data:raise RuntimeError('Immutable code path changed')
p.write_bytes(data)
r=Path('/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021')
health=json.loads((root/'training_fp64_sdpa_v4/health_rank0.json').read_text())
assert health['status']=='TRAINING'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
assert int(gpu.strip().splitlines()[1].split(',')[1])>10000
manifest=json.loads((root/'training_fp64_sdpa_v4/checkpoint_ready_000013000.json').read_text())
assert manifest['optimizer_updates']==13001
for item in manifest['files']:
 q=Path(item['file']);h=__import__('hashlib').sha256()
 with q.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
 assert q.stat().st_size==item['bytes'] and h.hexdigest()==item['sha256']
cmd=[str(r/'env/bin/python'),'-u',str(p),'--runtime-root',str(r),'--code-root',str(r/'code'),
 '--checkpoint',str(root/'training_fp64_sdpa_v4/checkpoints/model000013000.pt'),
 '--output',str(root/'reactive_async_v8_13001'),'--steps','500','--prompt','walk forward','--compile-planner']
env=os.environ.copy();env.update(TORCHINDUCTOR_CACHE_DIR='/home/ouyanghao/paper-faithful-compile-cache-20261022',TORCHINDUCTOR_COMPILE_THREADS='2',CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',MUJOCO_GL='egl')
if (root/'async_launch_v8_13001.json').exists():raise RuntimeError('Run already launched')
supervisor='import subprocess,json,sys,datetime; r=subprocess.run(sys.argv[2:]); open(sys.argv[1],"w").write(json.dumps({{"exit_code":r.returncode,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}})); sys.exit(r.returncode)'
with (root/'async_v8_13001.stdout.txt').open('wb') as out,(root/'async_v8_13001.stderr.txt').open('wb') as err:
 process=subprocess.Popen([str(r/'env/bin/python'),'-c',supervisor,str(root/'async_v8_13001_exit.json'),*cmd],stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True,env=env,cwd=str(r/'code'))
receipt={{'supervisor_pid':process.pid,'command':cmd,'source_sha256':{sha(source.read_bytes())!r},'free_data_bytes':shutil.disk_usage(root).free}}
(root/'async_launch_v8_13001.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''
result=json.loads(remote(script));dump(TASK/'async_launch_v8_13001.json',result);dump(BASE/'results/reactive_async_launch_v8_13001.json',result);print(json.dumps(result))
