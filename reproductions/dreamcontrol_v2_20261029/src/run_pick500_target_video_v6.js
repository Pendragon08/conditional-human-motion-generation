const fs=require('fs'),path=require('path'),{execFileSync}=require('child_process');
const filename='stage3_pick500_target_video_v6.py',L=__dirname;
let code=fs.readFileSync(path.join(L,'stage3_pick500_target_video_v5.py'),'utf8').replaceAll('d.text(','draw_mixed(d,');
const helper=`
def draw_mixed(draw,xy,value,font,fill):
    # The installed CJK fallback font has no ASCII glyphs. Render ASCII runs
    # with the existing DejaVu face so time/error digits remain readable.
    from itertools import groupby
    x,y=xy
    latin=ImageFont.truetype(FONT_EN,font.size)
    for ascii_run,chars in groupby(value,key=lambda c:ord(c)<128):
        text=''.join(chars);face=latin if ascii_run else font
        draw.text((x,y),text,font=face,fill=fill,anchor='lt')
        x+=draw.textlength(text,font=face)

`;
code=code.replace('\ndef main():',helper+'\ndef main():');fs.writeFileSync(path.join(L,filename),code);
const dest='/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/stage3_pick29_20261103/src/';
execFileSync('scp',['-q','-o','BatchMode=yes','-o','ConnectTimeout=15',path.join(L,filename),'10.147.17.32:'+dest],{stdio:'inherit',timeout:30000});
const script=String.raw`import json,subprocess,ast
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
code=S/'src/stage3_pick500_target_video_v6.py';assert code.exists();ast.parse(code.read_text())
assert json.loads((O/'target_report_video_v5/report.json').read_text())['status']=='rendered_verified'
receipt=O/'target_report_video_v6_launch.json';assert not receipt.exists()
source=(O/'target_report_video_v5_guard.py').read_text().replace('target_report_video_v5','target_report_video_v6').replace('stage3_pick500_target_video_v5.py','stage3_pick500_target_video_v6.py')
guard=O/'target_report_video_v6_guard.py';guard.open('x').write(source)
with (O/'target_report_video_v6_guard_launcher.log').open('x') as f:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(guard)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt.open('x').write(json.dumps({'guard_pid':child.pid,'guard':str(guard),'fix':'use existing Latin and CJK faces together, preserving actual numeric overlay values'},indent=2));print(receipt.read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
