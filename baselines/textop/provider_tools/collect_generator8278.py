"""Read-only generator8278 progress; never launch/stop/copy checkpoints."""
import argparse,json,pathlib,sys
from session import remote
BASE=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();assert a.run.isidentifier() and a.run.isascii()
script="""import pathlib,json,subprocess,hashlib,shutil
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN
p=r/'generator8278_20261027/vae_runs'/run/'receipt.json';ep=r/'receipts'/('generator8278_vae_'+run+'_exit.json')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out={'run':run,'receipt':json.loads(p.read_text()) if p.exists() else None,'receipt_sha256':h(p) if p.exists() else None,'exit':json.loads(ep.read_text()) if ep.exists() else None,'exit_sha256':h(ep) if ep.exists() else None,'disk_free':{str(p):shutil.disk_usage(p).free for p in [r,pathlib.Path('/home/ouyanghao')]}}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],text=True)
out['workers']=[x for x in subprocess.check_output(['ps','-u','1013','-o','pid,ppid,lstart,stat,args'],text=True).splitlines() if 'generator8278' in x]
print(json.dumps(out))
""".replace('RUN',repr(a.run))
value=json.loads(remote(script,timeout=60));out=BASE/'results/generator8278_20261027'/a.run;out.mkdir(parents=True,exist_ok=True)
(out/'progress.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
r=value['receipt'] or {};print(json.dumps({'run':a.run,'status':r.get('status'),'attempts':r.get('completed_primitive_attempts'),'updates':r.get('optimizer_updates'),'exit':value['exit'],'disk_free':value['disk_free']}))
