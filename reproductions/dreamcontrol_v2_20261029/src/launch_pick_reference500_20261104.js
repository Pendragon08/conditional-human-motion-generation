const cp=require('child_process');
const mode=process.argv[2];if(!['pilot','remaining'].includes(mode))throw Error('pilot or remaining required');
const py=`import ast,json,os,subprocess,hashlib,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';D=R/'stage2_prior29_20261103';O=S/'reference_scale500_20261104'
mode='${mode}'
code=S/'src/stage3_sample_pick29_batch.py';ast.parse(code.read_text())
assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2
spec=json.loads((D/'env_spec.json').read_text())['spec']
config=json.loads((D/'config.json').read_text())
assert spec['torch']=='2.5.1+cu121'
if mode=='pilot':name='candidate_pilot_v1';start=0;count=4;seconds=600
else:
 pilot=O/'candidate_pilot_v1';result=json.loads((pilot/'report.json').read_text())
 assert result['status']=='sampling_completed_not_postoptimized' and result['generated_candidates']==4
 audit=json.loads((O/'pilot_candidate_audit.json').read_text());assert audit['passed']
 name='candidate_remaining496_v1';start=4;count=496;seconds=28800
output=O/name;receipt=O/(name+'_launch.json')
assert not output.exists() and not receipt.exists(),'already launched or outputs exist; inspect rather than restart'
env=os.environ.copy();env.pop('CUDA_VISIBLE_DEVICES',None)
env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=':'.join(spec['module_paths']))
command=['timeout','--kill-after=20s',str(seconds),spec['python_path'],'-u',str(code),
         '--output-dir',str(output),'--count',str(count),'--start-index',str(start),'--batch-size','2']
with (O/(name+'.log')).open('x') as log:
 process=subprocess.Popen(command,cwd=config['trajgen'],env=env,stdin=subprocess.DEVNULL,
                          stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':process.pid,'command':command,'source_sha256':hashlib.sha256(code.read_bytes()).hexdigest(),
       'existing_env_spec':'74af067a','count':count,'start_index':start,'gpu_fraction':.05,
       'purpose':'task-conditioned candidate generation only, not admitted references or PPO',
       'protected_disk_floor_bytes':8*1024**3+456*1024**2,'output':str(output),'formal_training_started':False}
receipt.open('x').write(json.dumps(value,indent=2));print(json.dumps(value))
`;
console.log(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:py,encoding:'utf8'}));
