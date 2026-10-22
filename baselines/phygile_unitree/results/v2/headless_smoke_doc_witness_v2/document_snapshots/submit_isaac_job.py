"""Submit a bounded isolated headless run on the user's existing server."""
from pathlib import Path
import argparse,json,re,subprocess
p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--name',required=True)
p.add_argument('--mode',choices=['train','preview','evaluate'],default='train');p.add_argument('--num-envs',type=int,default=64)
p.add_argument('--iterations',type=int,default=2);p.add_argument('--steps',type=int,default=3);p.add_argument('--motion-file')
p.add_argument('--checkpoint',default='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/logs/rsl_rl/phygile_moe_pilot/2026-10-17_11-25-01_extended2000_seed42/model_1999.pt')
a=p.parse_args()
if not re.fullmatch(r'code_v\d+',a.version) or not re.fullmatch(r'[a-zA-Z0-9_-]+',a.name):raise ValueError('Invalid immutable source/run name')
config=vars(a)
script='''from pathlib import Path
import subprocess,os,json,datetime
c=CONFIG;r=Path('/data/code/ouyanghao/oyh/phygile-unitree-20261022');code=r/c['version'];out=r/'runs'/c['name']
assert r.resolve().parent==Path('/data/code/ouyanghao/oyh') and r.stat().st_uid==os.getuid()
assert (code/'scripts/lab_run.py').is_file() and not out.exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
free={int(row.split(',')[0]):int(row.split(',')[1]) for row in gpu.strip().splitlines()}
assert free[1]>=6144,'Insufficient shared GPU headroom'
import shutil
assert shutil.disk_usage(r).free>=1024**3,'Insufficient disk reserve; no data may be deleted automatically'
args=['taskset','-c','16-19','bash',str(code/'scripts/run_isaac.sh'),str(code/'scripts/lab_run.py'),'--mode',c['mode'],'--checkpoint',c['checkpoint'],'--output',str(out),'--num-envs',str(c['num_envs']),'--iterations',str(c['iterations']),'--steps',str(c['steps']),'--device','cuda:1','--headless','--headless-authorization',str(code/'configs/headless_authorization_20261022.json'),'--kit_args=--/renderer/multiGpu/enabled=false --/renderer/activeGpu=1 --/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4']
if c['motion_file']:args.extend(['--motion-file',c['motion_file']])
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',WANDB_MODE='disabled');env.pop('DISPLAY',None)
with (r/'logs'/(c['name']+'_stdout.txt')).open('x') as stdout,(r/'logs'/(c['name']+'_stderr.txt')).open('x') as stderr:
 process=subprocess.Popen(args,cwd=r,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':process.pid,'command':args,'gpu_free_mib_before':free,'training':c['mode']=='train','output':str(out)}
(r/'logs'/(c['name']+'_launch.json')).write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''.replace('CONFIG',repr(config))
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i','C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32','python3 -']
r=subprocess.run(ssh,input=script,text=True,capture_output=True,encoding='utf-8',timeout=60)
print(r.stdout);print(r.stderr);raise SystemExit(r.returncode)
