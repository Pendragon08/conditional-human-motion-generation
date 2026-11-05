const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const filename='stage3_pick500_target_video_v5.py',L=__dirname;
let code=fs.readFileSync(path.join(L,'stage3_pick500_target_video_v4.py'),'utf8');
code=code.replaceAll('torch.tensor([eye],device=env.device)','torch.tensor([eye],device=env.device,dtype=torch.float32)')
 .replaceAll('torch.tensor([look],device=env.device)','torch.tensor([look],device=env.device,dtype=torch.float32)')
 .replaceAll('tuple(v)','tuple(map(float,v))').replaceAll('tuple(box[i])','tuple(map(float,box[i]))').replaceAll('tuple(box[j])','tuple(map(float,box[j]))')
 .replaceAll('x,y=point;','x,y=map(float,point);').replaceAll('x,y=trail[-1];gx,gy=path[f]','x,y=map(float,trail[-1]);gx,gy=map(float,path[f])');
fs.writeFileSync(path.join(L,filename),code);
const dest='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103/src/';
execFileSync('scp',['-q','-o','BatchMode=yes','-o','ConnectTimeout=15',path.join(L,filename),'10.147.17.32:'+dest],{stdio:'inherit',timeout:30000});
const script=String.raw`import json,subprocess,ast
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
code=S/'src/stage3_pick500_target_video_v5.py';assert code.exists();ast.parse(code.read_text())
old=json.loads((O/'target_report_video_v4/failure.json').read_text());assert 'dtype Double' in old['error']
receipt=O/'target_report_video_v5_launch.json';assert not receipt.exists()
source=(O/'target_report_video_v4_guard.py').read_text().replace('target_report_video_v4','target_report_video_v5').replace('stage3_pick500_target_video_v4.py','stage3_pick500_target_video_v5.py')
guard=O/'target_report_video_v5_guard.py';guard.open('x').write(source)
with (O/'target_report_video_v5_guard_launcher.log').open('x') as f:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(guard)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt.open('x').write(json.dumps({'guard_pid':child.pid,'guard':str(guard),'fix':'camera eye float32 to match native up-axis float32; failed outputs preserved'},indent=2));print(receipt.read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
