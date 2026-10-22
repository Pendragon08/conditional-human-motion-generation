"""Bounded actual Unitree execution/replay test, with original assets read-only."""
from pathlib import Path
import argparse,json,re,subprocess
p=argparse.ArgumentParser(allow_abbrev=False)
p.add_argument('--version',required=True);p.add_argument('--name',required=True)
p.add_argument('--checkpoint',default='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/logs/rsl_rl/phygile_moe_pilot/2026-10-17_11-25-01_extended2000_seed42/model_1999.pt')
a=p.parse_args()
assert re.fullmatch(r'code_v\d+',a.version) and re.fullmatch(r'[A-Za-z0-9_-]+',a.name)
script='''from pathlib import Path
import os,json,subprocess,datetime,shutil
c=CONFIG;r=Path('/data/code/ouyanghao/oyh/phygile-unitree-20261022');p=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017');code=r/c['version'];out=r/'runs'/c['name']
assert r.stat().st_uid==os.getuid() and not out.exists()
assert set(range(16,20)).issubset(os.sched_getaffinity(0)), 'Documented CPU affinity unavailable'
assert shutil.disk_usage(r).free>1024**3
g=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
free={int(x.split(',')[0]):int(x.split(',')[1]) for x in g.strip().splitlines()};assert free[1]>=6144
motion=sorted((p/'data/test').glob('*.npz'))[0]
args=['taskset','-c','16-19','bash',str(code/'scripts/run_isaac.sh'),str(code/'scripts/run_unitree_closed_loop.py'),'--checkpoint',c['checkpoint'],'--motion_file',str(motion),'--normalization',str(p/'results/completion_v1/document_run/codec/output/normalization.npz'),'--output',str(out),'--device','cuda:1','--total_frames','50','--attempts','2','--headless']
env=os.environ.copy();env.pop('DISPLAY',None);env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled')
with (r/'logs'/(c['name']+'_stdout.txt')).open('x') as so,(r/'logs'/(c['name']+'_stderr.txt')).open('x') as se:
 q=subprocess.Popen(['python3',str(code/'scripts/supervise_run.py'),'--record',str(r/'logs'/(c['name']+'_exit.json')),'--']+args,cwd=r,env=env,stdout=so,stderr=se,start_new_session=True)
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':q.pid,'command':args,'gpu_free_mib_before':free,'output':str(out),'trained_generator':False,'semantic_generation_claim':False}
(r/'logs'/(c['name']+'_launch.json')).write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''.replace('CONFIG',repr(vars(a)))
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32','python3 -']
r=subprocess.run(ssh,input=script,text=True,capture_output=True,encoding='utf-8',timeout=60)
print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
