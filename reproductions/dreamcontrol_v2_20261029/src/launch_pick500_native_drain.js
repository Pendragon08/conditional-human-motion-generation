const cp=require('child_process');
const py=`import ast,json,os,subprocess,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
registry=O/'native_pipeline_registry.json'
if not registry.exists():
 value={'producer_dirs':[str(O/'candidate_pilot_v1'),str(O/'candidate_remaining496_v1')],
  'optimizer_record_paths':[str(O/'postopt_remaining496_jit_v2/records.jsonl'),str(O/'postopt_remaining488_jit_v3/records.jsonl')],
  'optimizer_terminal_dirs':[str(O/'postopt_pilot_jit_v1'),str(O/'postopt_remaining488_jit_v3')],
  'pre_v3_rejected_ids':[11],
  'active_jobs':[{'pid':1402368,'script':'stage3_sample_pick29_batch.py'},{'pid':1405357,'script':'stage3_pick500_postopt_follow_v3.py'}]}
 registry.open('x').write(json.dumps(value,indent=2))
for name in ['stage3_pick500_native_batch_v2.py','stage3_pick500_native_drain.py']:ast.parse((S/'src'/name).read_text())
assert not (O/'native_drain_controller_v1').exists()
assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2+32*1024**2
cmd=['nice','-n','10','timeout','--kill-after=20s','12500','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python',
 '-u',str(S/'src/stage3_pick500_native_drain.py')]
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (O/'native_drain_controller_v1.log').open('x') as log:
 child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':child.pid,'command':cmd,'scope':'data-only controlled native admission batches; not policy training',
 'resource_limits':{'combined_owned_GPU_bytes':12*1024**3,'combined_owned_RSS_bytes':32*1024**3,'disk_floor_bytes':8*1024**3+456*1024**2},
 'stops_when':'500 native references or available sources drained requiring replacement, then independent final-library checks'}
(O/'native_drain_controller_v1_launch.json').open('x').write(json.dumps(receipt,indent=2));print(json.dumps(receipt))
`;
console.log(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:py,encoding:'utf8'}));
