const {execFileSync}=require('child_process');
const script=String.raw`import json,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');O=R/'stage3_pick29_20261103/reference_scale500_20261104'
receipt=O/'ppo_integration_v1_launch.json';assert not receipt.exists(),'inspect existing integration receipt'
source=(O/'report_video_v1_guard.py').read_text()
source=source.replace('out=O/"report_video_v1"','out=O/"ppo_integration_v1"')
lines=source.splitlines()
lines=[('cmd=[spec["python"],"-u",str(S/"src/stage3_pick500_ppo.py"),"--headless","--device","cuda:1","--reference-manifest",manifest,"--output-dir",str(out),"--integration-test","--num-envs","4","--max-iterations","2"]' if x.startswith('cmd=') else x) for x in lines]
source='\n'.join(lines)+'\n'
source=source.replace('report_video_v1_guard.json','ppo_integration_v1_guard.json').replace('report_video_v1.log','ppo_integration_v1.log').replace('rendering','bounded_ppo_integration').replace('render_complete','integration_completed').replace('reference pose playback, not policy or grasp success','4env2updates on the500-scene adapter, not formal training or task-success proof')
guard=O/'ppo_integration_v1_guard.py';guard.open('x').write(source)
with (O/'ppo_integration_guard_launcher.log').open('x') as f:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(guard)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt.open('x').write(json.dumps({'guard_pid':child.pid,'guard':str(guard),'scope':'bounded4env2update integration; no formal run'},indent=2))
print(receipt.read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
