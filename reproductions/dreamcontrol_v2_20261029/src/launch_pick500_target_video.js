const {execFileSync}=require('child_process');
const script=String.raw`import json,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');O=R/'stage3_pick29_20261103/reference_scale500_20261104'
receipt=O/'target_report_video_v2_launch.json';assert not receipt.exists(),'inspect existing target-video receipt'
source=(O/'report_video_v1_guard.py').read_text()
source=source.replace('out=O/"report_video_v1"','out=O/"target_report_video_v2"')
lines=source.splitlines()
lines=[('cmd=[spec["python"],"-u",str(S/"src/stage3_pick500_target_video.py"),"--headless","--enable_cameras","--device","cuda:1","--original-report",str(O/"report_video_v1/report.json"),"--output-dir",str(out)]' if x.startswith('cmd=') else x) for x in lines]
source='\n'.join(lines)+'\n'
source=source.replace('report_video_v1_guard.json','target_report_video_v2_guard.json').replace('report_video_v1.log','target_report_video_v2.log')
guard=O/'target_report_video_v2_guard.py';guard.open('x').write(source)
with (O/'target_report_video_v2_guard_launcher.log').open('x') as f:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(guard)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt.open('x').write(json.dumps({'guard_pid':child.pid,'guard':str(guard),'scope':'same three generated references with native wrist goals and measured errors; no grasp-success animation'},indent=2))
print(receipt.read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
