const {execFileSync}=require('child_process');
const script=String.raw`import json,subprocess,ast
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
code=S/'src/stage3_pick500_target_video_v4.py';assert code.exists();ast.parse(code.read_text())
old=json.loads((O/'target_report_video_v3_guard.json').read_text());assert old['status']=='failed_preserved'
receipt=O/'target_report_video_v4_launch.json';assert not receipt.exists()
source=(O/'target_report_video_v3_guard.py').read_text().replace('target_report_video_v3','target_report_video_v4').replace('stage3_pick500_target_video_v3.py','stage3_pick500_target_video_v4.py')
guard=O/'target_report_video_v4_guard.py';guard.open('x').write(source)
with (O/'target_report_video_v4_guard_launcher.log').open('x') as f:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(guard)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt.open('x').write(json.dumps({'guard_pid':child.pid,'guard':str(guard),'scope':'diagnostic run; earlier failed artifacts preserved'},indent=2));print(receipt.read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
