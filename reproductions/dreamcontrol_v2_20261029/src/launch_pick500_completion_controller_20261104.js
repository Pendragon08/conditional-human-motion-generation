const {execFileSync}=require('child_process');
const script=String.raw`import ast,json,subprocess,shutil,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104';Q=R/'stage3_skill_sequence_20261103'
source=S/'src/stage3_pick500_completion_controller.py'
for name in ['stage3_pick500_completion_controller.py','stage3_pick500_finalize_library.py','stage3_pick500_evaluate.py','stage3_pick500_ppo.py']:ast.parse((S/'src'/name).read_text())
assert json.loads((Q/'sequence.json').read_text())['active_task_order']==['pick','pour','bimanual_box_lift','box_push_pull']
assert json.loads((O/'evaluation_entry4_v1/report.json').read_text())['status']=='integration_evaluation_completed'
assert json.loads((O/'replacement_queue128_v1/status.json').read_text())['status']=='replacement_sources_running'
assert shutil.disk_usage('/data').free>8*1024**3+456*1024**2
receipt=O/'completion_controller_v1_launch.json';assert not receipt.exists() and not (O/'completion_controller_v1').exists()
with (O/'completion_controller_v1.log').open('x') as log:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(source)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value={'controller_pid':child.pid,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'status':'waiting_reference500_not_formal_training_started','data_floor':8*1024**3+456*1024**2,'training_start_floor':11*1024**3,
 'stages':['wait native500 and source audits','CPU final manifest recomputation','actual500-bank8912env128step capacity','resource-gated newPick3000PPO','100envread-only evaluation/archive'],
 'no_old_baseline_restart':True,'no_resource_floor_reduction':True}
receipt.open('x').write(json.dumps(value,indent=2))
active=O/'active_completion_controller.json';assert not active.exists();active.open('x').write(json.dumps(value,indent=2));print(json.dumps(value))
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
