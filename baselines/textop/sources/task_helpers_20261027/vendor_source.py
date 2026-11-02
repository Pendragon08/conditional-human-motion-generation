import pathlib,sys,subprocess,shutil,json,hashlib
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import REPO
BASE=REPO/'baselines/textop';up=TASK/'upstream'
tracked=subprocess.run(['git','-C',str(up),'ls-files','-z'],capture_output=True,check=True).stdout.decode().split('\0')
receipt=[]
for name in filter(None,tracked):
    src=up/name
    if not src.is_file() or src.is_symlink():continue
    dst=BASE/'upstream'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    receipt.append({'path':name,'bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
meta={'repo':'https://github.com/TeleHuman/TextOp','commit':subprocess.run(['git','-C',str(up),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),'files':receipt,'unmodified_official_source':True,'submodules':'ROS2 deployment submodules not populated in this training stage'}
(BASE/'sources/official_source_receipt.json').write_text(json.dumps(meta,indent=2))
shutil.copy2(BASE/'sources/hf_asset_manifest.json',BASE/'configs/asset_manifest.json')
(BASE/'.gitignore').write_text('env/\nassets/\nruns/\nreceipts/\nlogs/\n__pycache__/\n*.partial\n*.partial.*\n')
print('VENDORED',len(receipt),sum(x['bytes'] for x in receipt),'LARGE',[(x['path'],x['bytes']) for x in receipt if x['bytes']>10*1024**2])
