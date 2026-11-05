const {execFileSync}=require('child_process');
const script=String.raw`import json,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');O=R/'stage3_pick29_20261103/reference_scale500_20261104'
old=O/'target_report_video_v2_guard.json';j=json.loads(old.read_text());assert j['returncode']==0 and not (O/'target_report_video_v2/report.json').exists()
j['status']='failed_output_verification_preserved';j['reason']='process exited0 but no completed report or valid video; do not call complete';old.write_text(json.dumps(j,indent=2))
receipt=O/'target_report_video_v3_launch.json';assert not receipt.exists()
source=(O/'target_report_video_v2_guard.py').read_text().replace('target_report_video_v2','target_report_video_v3').replace('stage3_pick500_target_video.py','stage3_pick500_target_video_v3.py')
source=source.replace('code=child.wait();state["returncode"]=code','code=child.wait();state["returncode"]=code\n if code==0 and not (out/"report.json").exists():state["status"]="failed_output_verification_preserved"')
guard=O/'target_report_video_v3_guard.py';guard.open('x').write(source)
with (O/'target_report_video_v3_guard_launcher.log').open('x') as f:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(guard)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt.open('x').write(json.dumps({'guard_pid':child.pid,'guard':str(guard),'scope':'new diagnostic attempt; failed v2 preserved'},indent=2))
print(receipt.read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
