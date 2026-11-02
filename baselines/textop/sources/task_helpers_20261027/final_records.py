import pathlib,sys,subprocess,json,hashlib
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import remote,REPO
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
print(remote(f"import pathlib,json; p=pathlib.Path('{ROOT}/reports'); p.mkdir(exist_ok=True); print(json.dumps({{'reports':str(p)}}))"))
files=[BASE/'ENVIRONMENT.md',BASE/'SERVER_NOTES.md',BASE/'reports/启动报告_20261024.md',BASE/'results/initial_checkpoint_backup.json',BASE/'results/checkpoint_sync_state.json',LOCAL/'TextOp下载与训练报告.html']
for p in files:subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519',str(p),'ouyanghao@10.147.17.32:'+ROOT+'/reports/'+p.name],check=True)
state=json.loads((BASE/'results/checkpoint_sync_state.json').read_text(encoding='utf-8'))
assert not state.get('last_error');assert state['last_inventory']['alive']
print('SYNC_HEALTHY',state['updated_utc'],'ARCHIVED_SERVER_COPIES',len(state['server_archives']))
print('LOCAL_REPORT_BYTES',(LOCAL/'TextOp下载与训练报告.html').stat().st_size)
