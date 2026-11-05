const {execFileSync}=require('child_process');
const script=String.raw`import json,hashlib
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');D=R/'stage2_prior29_20261103'
print('CONFIG',json.dumps(json.loads((D/'config.json').read_text()),ensure_ascii=False))
for d in ['grab_omni_mixed29_full_preview_20261103','grab_omni_mixed29_dataset_20261103','stage2_prior29_20261103/data']:
 p=R/d/'manifest.json'
 if p.exists():
  j=json.loads(p.read_text());print('MANIFEST',str(p),hashlib.sha256(p.read_bytes()).hexdigest(),list(j));print('ROW',json.dumps(j.get('records',[{}])[0],ensure_ascii=False))
print('STATUS',json.dumps(json.loads((D/'runs/grab_omni29_mdm_baseline_v1/status.json').read_text()),ensure_ascii=False)[:4000])
`;
process.stdout.write(execFileSync('ssh',['-o','BatchMode=yes','-o','ConnectTimeout=15','10.147.17.32','python3 -'],{input:script,encoding:'utf8',timeout:30000}));
