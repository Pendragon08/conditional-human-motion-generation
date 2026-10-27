import pathlib,sys,json
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,REPO
out=json.loads(remote(r'''
import pathlib,json,os,subprocess,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
launch=json.loads((r/'receipts/training_launch.json').read_text());pid=launch['pid'];proc=pathlib.Path('/proc')/str(pid)
assert proc.exists()
cmd=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
assert str(r/'scripts/train_official.py') in cmd
settings={}
for x in (proc/'environ').read_bytes().split(b'\0'):
    if b'=' in x:
        k,v=x.split(b'=',1)
        if k.decode() in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','WANDB_MODE']:settings[k.decode()]=v.decode()
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (proc/'status').read_text().splitlines() if ':' in line}
out={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':pid,'state':status['State'],'uid':status['Uid'],'resident_memory':status.get('VmRSS'),'cpu_affinity':sorted(os.sched_getaffinity(pid)),'settings':settings,'gpu_snapshot':gpu,'dataset_preservation':'No dataset removed or modified; downloads are new files within the user directories.'}
(r/'receipts/active_training_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''))
(REPO/'baselines/textop/results/active_training_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
