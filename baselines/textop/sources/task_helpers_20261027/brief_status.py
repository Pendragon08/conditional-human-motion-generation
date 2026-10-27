import pathlib,json,shutil,time
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=r/'assets/TextOpRobotMDAR/dataset/BABEL-AMASS-ROBOT-23dof-FULL-50fps/train.pkl.partial.ranges.json'
if p.exists():
    s=json.loads(p.read_text());print('TRAIN_RANGE_PROGRESS',s['prefix']+len(s['completed'])*16777216,'/',4905919425)
print('TIME',time.strftime('%H:%M:%S',time.gmtime()))
for n in ['download_complete.json','training_launch.json','training_progress.json','training_exit.json','numerical_failure.json','disk_guard_stop.json']:
    p=r/'receipts'/n
    if p.exists():
        j=json.loads(p.read_text());print(n,{'count':len(j.get('files',[])),'bytes':j.get('total_bytes')} if n=='download_complete.json' else j)
for n in ['mvae.log','gate.log','download.log']:
    p=r/'logs'/n
    if p.exists():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-800));print(n,f.read().decode(errors='replace'))
print('FREE_GIB',shutil.disk_usage(r).free/1024**3)
