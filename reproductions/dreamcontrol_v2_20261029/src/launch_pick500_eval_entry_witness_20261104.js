const {execFileSync}=require('child_process');
const script=String.raw`import json,ast,subprocess
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104'
source=S/'src/stage3_pick500_evaluate.py';ast.parse(source.read_text())
receipt=O/'evaluation_entry4_v1_launch.json';assert not receipt.exists() and not (O/'evaluation_entry4_v1').exists()
base=(O/'report_video_v1_guard.py').read_text();base=base.replace('out=O/"report_video_v1"','out=O/"evaluation_entry4_v1"')
lines=base.splitlines();lines=[('manifest=json.loads((O/"ppo_integration_v1/provenance.json").read_text())["reference_bank_sha256"]\nmanifest_path=O/"native_batches/batch_0008/partial_reference_bank.json"' if x.startswith('manifest=') else x) for x in lines]
cmd='cmd=[spec["python"],"-u",str(S/"src/stage3_pick500_evaluate.py"),"--headless","--device","cuda:1","--reference-manifest",str(manifest_path),"--training-run",str(O/"ppo_integration_v1"),"--checkpoint",str(O/"ppo_integration_v1/ppo/model_1.pt"),"--output-dir",str(out),"--integration-test","--num-envs","4","--steps","32"]'
lines=[cmd if x.startswith('cmd=') else x for x in lines];base='\n'.join(lines)+'\n'
base=base.replace('report_video_v1_guard.json','evaluation_entry4_v1_guard.json').replace('report_video_v1.log','evaluation_entry4_v1.log').replace('rendering','evaluating_entry_test').replace('render_complete','evaluation_entry_completed')
base=base.replace('code=child.wait();state["returncode"]=code','code=child.wait();state["returncode"]=code\n if code==0 and (not (out/"report.json").exists() or json.loads((out/"report.json").read_text())["status"]!="integration_evaluation_completed"):state["status"]="failed_output_verification_preserved"')
guard=O/'evaluation_entry4_v1_guard.py';guard.open('x').write(base)
with (O/'evaluation_entry4_v1_guard_launcher.log').open('x') as f:
 child=subprocess.Popen(['nice','-n','10','python3','-u',str(guard)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
receipt.open('x').write(json.dumps({'guard_pid':child.pid,'guard':str(guard),'scope':'new evaluator checkpoint/liveActor/fresh-target4env32step witness; no learning or final task-success result'},indent=2));print(receipt.read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
