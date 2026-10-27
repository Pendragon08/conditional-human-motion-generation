import sys,pathlib,json,subprocess,os
TASK=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(TASK.parent/'paper_faithful_reproduction_20261022'))
from session import REPO,SSH,GH
BASE=REPO/'baselines/textop';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024');ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
PY=sys.executable
progress=json.loads((BASE/'results/training_progress.json').read_text())
launch=json.loads((BASE/'results/training_launch.json').read_text())
c={'baseline':str(BASE),'local':str(LOCAL),'proxy':'http://127.0.0.1:7897','ssh':SSH,'scp':['scp','-i','C:/Users/Pende/.ssh/id_ed25519'],'host':'ouyanghao@10.147.17.32','gh':GH,'github_repo':'Pendragon08/conditional-human-motion-generation','server_root':ROOT,'run_dir':progress['save_dir'],'server_python':ROOT+'/env/bin/python','training_pid':launch['pid'],'local_python':PY,'orchestrator':str(BASE/'provider_tools'),'scope':'This single TextOp MVAE run only; retain all checkpoint versions locally and GitHub; retain newest verified checkpoint on server; no dataset deletion'}
cfg=BASE/'configs/checkpoint-sync.local.json';cfg.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
command=[PY,str(BASE/'scripts/checkpoint_sync_local.py'),'--config',str(cfg)]
if '--start' in sys.argv:
    assert not (TASK/'checkpoint_sync_pid.json').exists()
    p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=open(TASK/'checkpoint_sync.log','ab'),stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    (TASK/'checkpoint_sync_pid.json').write_text(json.dumps({'pid':p.pid,'command':command},ensure_ascii=False,indent=2),encoding='utf-8');print('CHECKPOINT_SYNC_PID',p.pid)
else:subprocess.run(command+['--once'],check=True)
