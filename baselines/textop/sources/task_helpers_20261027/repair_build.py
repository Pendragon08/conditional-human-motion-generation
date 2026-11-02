import sys,pathlib,json,hashlib,subprocess
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,REPO
BASE=REPO/'baselines/textop';ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
specpath=BASE/'configs/env-spec.json'
spec=json.loads(specpath.read_text());spec['pip_phases'][2][1]='-e upstream/deps/isaac_utils --no-deps'
specpath.write_text(json.dumps(spec,indent=2))
spec_hash=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
with (BASE/'SERVER_NOTES.md').open('a') as f:f.write(f'\n### env: textop@{spec_hash}\nSpec corrected: isaac_utils is at upstream/deps/isaac_utils, not under TextOpRobotMDAR. Original failed build log retained. Rebuild pending.\n')
subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519',str(specpath),'ouyanghao@10.147.17.32:'+ROOT+'/configs/env-spec.json'],check=True)
print(remote(f'''
import pathlib,subprocess,os,json
r=pathlib.Path('{ROOT}')
(r/'logs/build.log').rename(r/'logs/build_attempt1_failed.log')
p=subprocess.Popen(['python3',str(r/'scripts/build_environment.py')],cwd=r,stdout=open(r/'logs/build.log','ab'),stderr=subprocess.STDOUT,start_new_session=True)
print('REBUILD_PID',p.pid)
'''))
