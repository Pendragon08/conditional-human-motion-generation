const cp=require('child_process');
const mode=process.argv[2]||'pilot';if(!['pilot','remaining'].includes(mode))throw Error('pilot or remaining required');
const py=`import ast,json,os,subprocess,hashlib,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103'
O=S/'reference_scale500_20261104';mode='${mode}'
if mode=='pilot':name='postopt_pilot_jit_v1';source=S/'src/stage3_pick500_postopt.py';records=O/'candidate_pilot_v1/records.jsonl';count=4;seconds=600
else:
 name='postopt_remaining496_jit_v2';source=S/'src/stage3_pick500_postopt_follow_v2.py';records=O/'candidate_remaining496_v1/records.jsonl';count=496;seconds=28800
 proof=json.loads((O/'native_pilot_v1_launch/report.json').read_text());assert proof['status']=='native_replay_and_object_reset_passed'
ast.parse(source.read_text())
old=S/'src/stage3_postopt_queue.py';tree=ast.parse(old.read_text())
assign=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='env' for t in x.targets))
ns={'R':R,'S':S,'os':os,'Path':Path}
exec(compile(ast.Module(body=[assign],type_ignores=[]),str(old),'exec'),ns)
env=ns['env'];env['CUDA_VISIBLE_DEVICES']='1';env['PYTHONDONTWRITEBYTECODE']='1'
assert env['XLA_PYTHON_CLIENT_MEM_FRACTION']=='0.05'
assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2
out=O/name;receipt=O/(name+'_launch.json')
assert not out.exists() and not receipt.exists()
command=['timeout','--kill-after=20s',str(seconds),'/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python',
 '-u',str(source),'--records',str(records),'--output-dir',str(out),
 '--initial-state',str(S/'reports/pick_table_bounded_0_v1/robot29.npz'),'--max-candidates',str(count)]
if mode=='remaining':command.append('--follow')
with (O/(name+'.log')).open('x') as log:
 child=subprocess.Popen(command,env=env,cwd=str(S),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':child.pid,'command':command,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'existing_environment_source':str(old),'existing_environment_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
 'physical_gpu':1,'jax_memory_fraction':.05,'software_installations':[],'purpose':'compiled generated-candidate postoptimization, not formal training','expected_count':count,'follow_producer':mode=='remaining'}
receipt.open('x').write(json.dumps(value,indent=2));print(json.dumps(value))
`;
console.log(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:py,encoding:'utf8'}));
