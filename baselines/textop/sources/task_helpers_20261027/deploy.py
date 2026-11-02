import sys,pathlib,subprocess,json
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,REPO
BASE=REPO/'baselines/textop'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
for p in (BASE/'scripts').glob('*.py'):
    subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519',str(p),'ouyanghao@10.147.17.32:'+ROOT+'/scripts/'+p.name],check=True)
if '--start' in sys.argv:
    print(remote(f'''
import subprocess,pathlib,json,urllib.request
r=pathlib.Path('{ROOT}')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({{'https':'http://127.0.0.1:17897'}}))
with opener.open('https://huggingface.co/api/datasets/Yochish/TextOp-Data',timeout=20) as q: print('PROXY_STATUS',q.status)
processes={{}}
for name,script in [('download','download_assets.py'),('build','build_environment.py')]:
    p=subprocess.Popen(['python3',str(r/'scripts'/script)],cwd=str(r),stdout=open(r/'logs'/(name+'.log'),'ab'),stderr=subprocess.STDOUT,start_new_session=True)
    processes[name]=p.pid
(r/'receipts/preparation_pids.json').write_text(json.dumps(processes))
print(json.dumps(processes))
''',timeout=50))
