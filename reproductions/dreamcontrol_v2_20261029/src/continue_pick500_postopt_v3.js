const cp=require('child_process');
const py=`import ast,json,os,subprocess,hashlib,shutil
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
previous=O/'postopt_remaining496_jit_v2';prefix=previous/'records.jsonl'
rows=[json.loads(x) for x in prefix.read_text().splitlines()]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert [r['candidate_id'] for r in rows]==list(range(4,11))
for r in rows:assert r['passed_requested_kinematic_checks'] and sha(r['output'])==r['output_sha256'] and sha(r['sample'])==r['sample_sha256']
assert 'AssertionError: bound correction >=1mrad' in (O/'postopt_remaining496_jit_v2.log').read_text()
proc=Path('/proc/1404446/stat')
if proc.exists():assert proc.read_text().split()[2]=='Z','old process still alive; do not relaunch'
raw=previous/'candidate_000011/solver_raw.npz';assert raw.exists()
reject={'candidate_id':11,'status':'rejected_preserved_raw_solver',
 'rejection_reason':'raw right_wrist_pitch_joint exceedsURDF upper bound by0.001222407rad; required projection exceeds original1mrad rule',
 'maximum_raw_violation_rad':.001222407141296289,'threshold_exclusive_rad':.001,
 'solver_raw_sha256':sha(raw),'preserved_prefix_ids':[r['candidate_id'] for r in rows],
 'eligible_for_calibrated_scene_reference':False,'next_action':'new seed replacement later, do not repeat this solve'}
receipt=O/'rejection_candidate11_v2.json'
if not receipt.exists():receipt.open('x').write(json.dumps(reject,indent=2))
f=previous/'failure_observed.json'
if not f.exists():f.open('x').write(json.dumps({'status':'failed_at_candidate11_preserved7complete','rejection_receipt':str(receipt),'source_log':str(O/'postopt_remaining496_jit_v2.log')},indent=2))
source=S/'src/stage3_pick500_postopt_follow_v3.py';ast.parse(source.read_text())
old=S/'src/stage3_postopt_queue.py';tree=ast.parse(old.read_text())
assign=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='env' for t in x.targets))
ns={'R':R,'S':S,'os':os,'Path':Path};exec(compile(ast.Module(body=[assign],type_ignores=[]),str(old),'exec'),ns)
env=ns['env'];env['CUDA_VISIBLE_DEVICES']='1';env['PYTHONDONTWRITEBYTECODE']='1'
assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2
name='postopt_remaining488_jit_v3';out=O/name;launch=O/(name+'_launch.json')
assert not out.exists() and not launch.exists(),'already launched or output exists'
command=['timeout','--kill-after=20s','28800','/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python',
 '-u',str(source),'--records',str(O/'candidate_remaining496_v1/records.jsonl'),'--output-dir',str(out),
 '--initial-state',str(S/'reports/pick_table_bounded_0_v1/robot29.npz'),'--max-candidates','488','--follow',
 '--exclude-records',str(prefix),'--exclude-ids','11']
with (O/(name+'.log')).open('x') as log:
 child=subprocess.Popen(command,env=env,cwd=str(S),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'pid':child.pid,'command':command,'source_sha256':sha(source),'preserved_completed_prefix':str(prefix),
 'excluded_rejected_ids':[11],'expected_new_processed':488,'physical_gpu':1,'jax_memory_fraction':.05,
 'scope':'continue only unprocessed generated candidates; per-candidate bound failures rejected without widening threshold','formal_training_started':False}
launch.open('x').write(json.dumps(value,indent=2));print(json.dumps(value))
`;
console.log(cp.execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:py,encoding:'utf8'}));
