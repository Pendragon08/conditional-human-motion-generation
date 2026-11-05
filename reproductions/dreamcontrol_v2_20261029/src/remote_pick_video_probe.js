const {execFileSync}=require('child_process');
const p='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python';
const script=`import importlib.util as u,json
from pathlib import Path
print(json.dumps({m:bool(u.find_spec(m)) for m in ['mujoco','pyrender','trimesh','matplotlib','PIL','cv2','OpenGL']}))
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
O=R/'stage3_pick29_20261103/reference_scale500_20261104'
j=json.loads((O/'library_progress.json').read_text());m=json.loads(Path(j['current_partial_manifest']).read_text());print(json.dumps(m['references'][0],indent=2))
print((R/'stage3_skill_sequence_20261103/sequence.json').read_text())
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32',p+' -'],{input:script,encoding:'utf8',timeout:30000}));
