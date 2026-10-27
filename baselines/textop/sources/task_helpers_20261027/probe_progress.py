import pathlib,json,shutil,subprocess
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
for path in [r/'assets',pathlib.Path('/home/ouyanghao/textop-assets-20261024')]:
    parts=[]
    for p in path.rglob('*.partial.ranges.json'):
        s=json.loads(p.read_text());parts.append({'path':p.name,'prefix':s['prefix'],'chunks':len(s['completed']),'approx_downloaded':s['prefix']+len(s['completed'])*16*1024**2})
    print('RANGE_PROGRESS',parts)
for name in ['download.log','env_finalize.log','mvae.log','gate.log']:
    p=r/'logs'/name
    if p.exists():
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2000));print(name,f.read().decode(errors='replace'))
for name in ['kernel_witness.json','training_progress.json','numerical_failure.json','training_failure.txt']:
    p=r/'receipts'/name
    if p.exists():print(name,p.read_text()[-1800:])
print('FREE_GiB',*[round(shutil.disk_usage(p).free/1024**3,3) for p in [r,pathlib.Path('/home/ouyanghao')]])
print(subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.free','--format=csv,noheader'],capture_output=True,text=True).stdout)
