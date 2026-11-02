"""Stage task-local runtime files without changing shared packages/assets."""
import hashlib,json,pathlib,subprocess,sys
from session import REPO,remote
BASE=REPO/'baselines/textop';ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
files=['configs/tracker-env-spec.json','scripts/run_tracker.sh','scripts/tracker_runtime_inventory.py']
spec=json.loads((BASE/files[0]).read_text(encoding='utf-8'))
tag=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
with (BASE/'SERVER_NOTES.md').open('a',encoding='utf-8') as f:
    f.write('\n### env: textop-tracker@'+tag+'\nSpec written before dispatch. Warm read-only IsaacSim4.5.0/IsaacLab2.1.0 reuse, pinned TextOp source injected in task PYTHONPATH. Shared assets and packages unchanged; GPU1, CPU4, Torch65% cap. Headless explicitly authorized. Package inventory and actual physics/FK witness pending; do not launch filter training yet.\n')
for rel in files:
    subprocess.run(['scp','-q','-o','BatchMode=yes','-i','C:/Users/Pende/.ssh/id_ed25519',str(BASE/rel),'ouyanghao@10.147.17.32:'+ROOT+'/'+rel],check=True)
print(remote("import subprocess,json,pathlib; r=pathlib.Path('"+ROOT+"');p=subprocess.run(['bash',str(r/'scripts/run_tracker.sh'),str(r/'scripts/tracker_runtime_inventory.py')],capture_output=True,text=True,timeout=90);print(p.stdout);print(p.stderr[-2000:]);print('EXIT',p.returncode)",timeout=110))
