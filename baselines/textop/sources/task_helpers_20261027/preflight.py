import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'paper_faithful_reproduction_20261022'))
from session import remote
script=r'''
import pathlib,os,json,shutil,subprocess,urllib.request
root=pathlib.Path('/data/code/ouyanghao/oyh')
r={'disks':{str(p):shutil.disk_usage(p)._asdict() for p in [root,pathlib.Path('/home/ouyanghao')]}}
r['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout
r['processes']=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid,args'],capture_output=True,text=True).stdout
r['existing_candidates']=[]
for base,dirs,files in os.walk(root):
    dirs[:]=[d for d in dirs if d not in ['site-packages','.git','node_modules','__pycache__','wandb'] and not d.startswith('.venv')]
    for d in dirs:
        if any(x in d.lower() for x in ['textop','babel-amass-robot']):r['existing_candidates'].append(str(pathlib.Path(base)/d))
try:
    with urllib.request.urlopen('https://huggingface.co/api/datasets/Yochish/TextOp-Data',timeout=20) as q:r['hf_status']=q.status
except Exception as e:r['hf_error']=repr(e)
print(json.dumps(r))
'''
result=json.loads(remote(script,timeout=100))
out=pathlib.Path('D:/具身智能/TextOp复现_20261024')
(out/'launch_preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='processes'},indent=2))
print(result['processes'][-2500:])
