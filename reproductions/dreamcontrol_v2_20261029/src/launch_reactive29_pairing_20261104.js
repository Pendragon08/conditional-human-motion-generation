const cp=require('child_process');
const py=`import ast,json,os,subprocess,hashlib,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');W=R/'reactive29_pairing_work_20261104'
source=W/'build_reactive29_paired_cpu_20261104.py';ast.parse(source.read_text())
out=W/'publisher60_20hz_full_v1';receipt=W/'publisher60_20hz_full_v1_launch.json'
assert not out.exists() and not receipt.exists(),'existing launch/output: inspect rather than restart'
assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2+1024**3
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
command=['nice','-n','10','timeout','--kill-after=20s','14400','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python',
 '-u',str(source),'--output-dir',str(out),'--publisher-contract',str(W/'reactive_publisher_data_contract_20261104.md')]
with (W/'publisher60_20hz_full_v1.log').open('x') as log:
 child=subprocess.Popen(command,cwd=str(W),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':child.pid,'command':command,'code_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'scope':'published11424 G1 motion-text reuse, nominal publisher60->20Hz, not human retargeting or training',
 'CPU_threads':1,'GPU_used':False,'priority_nice':10,'output_file_budget_bytes':1024**3,
 'disk_floor_bytes':8*1024**3+456*1024**2,'original_data_and_current_models_read_only':True}
receipt.open('x').write(json.dumps(value,indent=2));print(json.dumps(value))
`;
console.log(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:py,encoding:'utf8'}));
