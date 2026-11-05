import json, pathlib, subprocess, shutil
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'stage3_supermimic29_20261103'; N=D/'native_universal_20261104'
for x in ['smoke_author_v1_guard.json','smoke_author_v1/status.json','continuation_status.json','capacity_8912_v1_guard.json','benchmark_8912_v1_guard.json']:
 p=N/x
 if p.exists(): print(x,p.read_text()[-11000:])
for x in ['smoke_author_v1.log','smoke_author_v1/logs/train_log.csv']:
 p=N/x
 if p.exists(): print(x,'\n'.join(p.read_text().splitlines()[-10:]))
p=R/'stage3_pick29_20261103/runs/pick500_8912_3000_v1/status.json'
print('PPO',p.read_text()[-4500:]); print('DISK',shutil.disk_usage('/data')._asdict())
for x in ['native_tracker500_capacity_worker.py','native_tracker500_capacity_guard.py','native_tracker500_continuation.py']:
 p=D/'src'/x; print('SOURCE',x,p.stat().st_size if p.exists() else 'MISSING')
print(subprocess.check_output(['ps','-p','1458014,1458015,1436632','-o','pid,ppid,etimes,%cpu,rss,stat,args'],text=True))
