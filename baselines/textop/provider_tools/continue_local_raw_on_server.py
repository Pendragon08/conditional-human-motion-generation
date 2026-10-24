"""Finite continuation for the1653 selected local inputs, without GPU training.

Raw inputs remain durable on C: and in original D: archives. Server RAM is only
a working copy because /data is nearly full. Prepared output is archived to D:
before completion. No files are deleted and no paused experiment is resumed.
"""
import datetime,hashlib,json,os,pathlib,shlex,subprocess,sys,tarfile,time
from session import SSH,remote
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
sys.stderr.reconfigure(encoding='utf-8',errors='backslashreplace')
HERE=pathlib.Path(__file__).resolve().parent;BASE=HERE.parent
TASK=BASE.parents[3]/'textop_reproduction_20261024'
PRIVATE=TASK/'private_assets';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RAM='/dev/shm/ouyanghao_textop_20261025/local1653_v1'
STATE=BASE/'results/data_preparation_20261025/local1653_queue.json'
PY='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'

def state(**values):
    old=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    old.update(values,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    STATE.write_text(json.dumps(old,indent=2),encoding='utf-8');print(json.dumps(old),flush=True)

def collect_publish():
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    for name in ['collect_data_preparation.py','publish.py']:
        p=subprocess.run([sys.executable,str(HERE/name)],env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if p.returncode:raise RuntimeError(name+': '+p.stderr[-1500:])
        print(p.stdout[-500:],flush=True)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    state(status='WAITING_FOR_LOCAL_EXTRACTION_AND_PREVIOUS_RETARGETING',target=1653,ram_working_copy=True,source_durable='C private assets and D licensed original archives',physical_filter_executed=False)
    ready=PRIVATE/'indexed_raw_expansion_v1/ready.private.json'
    for _ in range(1440):
        p=PRIVATE/'raw_expansion_progress.json'
        progress=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        if progress.get('status')=='STOPPED_WITH_ERROR':raise RuntimeError('Local extraction stopped; retaining all files')
        if ready.exists():
            previous=json.loads(remote("import pathlib,json; p=pathlib.Path('"+ROOT+"/data_preparation_20261025/available143_v1/progress.json'); print(p.read_text() if p.exists() else '{}')",timeout=40))
            if previous.get('status')=='PREPROCESSING_FINISHED_FILTER_PENDING':
                assert previous['completed_sequences']==143 and previous['failed_sequences']==0;break
        time.sleep(30)
    else:raise TimeoutError('Prerequisites did not finish in12h; no new processing launched')
    rows=json.loads(ready.read_text(encoding='utf-8'));assert len(rows)==1653
    paths=[pathlib.Path(x['local_raw_path']) for x in rows]
    for row,path in zip(rows,paths):assert path.is_file() and sha(path)==row['raw_sha256']
    total=sum(p.stat().st_size for p in paths)
    preflight=f'''
import os,pathlib,json,shutil
assert os.getuid()==1013
root=pathlib.Path('{RAM}');parent=root.parent
if parent.exists():assert parent.stat().st_uid==1013
parent.mkdir(mode=0o700,exist_ok=True)
assert not root.exists(),'Existing RAM working copy preserved; refusing duplicate staging'
mem={{x.split(':')[0]:int(x.split()[1])*1024 for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemTotal:','MemAvailable:'))}}
assert mem['MemAvailable']>{total}*2+8*1024**3,mem
assert shutil.disk_usage('/dev/shm').free>{total}*3+8*1024**3
root.mkdir(mode=0o700);(root/'raw').mkdir(mode=0o700)
print(json.dumps({{'mem':mem,'raw_bytes':{total},'ram_root':str(root)}}))
'''
    proof=json.loads(remote(preflight));state(status='STAGING_VERIFIED_RAW_WORKING_COPY',preflight=proof)
    stderr=TASK/'ram1653_transfer.stderr.log'
    with stderr.open('xb') as error:
        proc=subprocess.Popen([*SSH,'tar -xzf - -C '+shlex.quote(RAM+'/raw')],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=error)
        try:
            with tarfile.open(fileobj=proc.stdin,mode='w|gz',compresslevel=1) as archive:
                for path in paths:archive.add(path,arcname=path.name,recursive=False)
            proc.stdin.close();assert proc.wait(timeout=120)==0,'Raw transfer failed; partial working copy preserved'
        except BaseException:
            proc.stdin.close();proc.wait(timeout=120);raise
    for row,path in zip(rows,paths):row['raw_stageii_candidates']=[RAM+'/raw/'+path.name]
    manifest=PRIVATE/'ram1653_pairs.private.json';manifest.write_text(json.dumps(rows),encoding='utf-8')
    subprocess.run(['scp','-i','C:/Users/Pende/.ssh/id_ed25519',str(manifest),'ouyanghao@10.147.17.32:'+RAM+'/pairs.private.json'],check=True,capture_output=True)
    launch=f'''
import pathlib,json,hashlib,subprocess,os,shutil,time
root=pathlib.Path('{RAM}');r=pathlib.Path('{ROOT}');rows=json.loads((root/'pairs.private.json').read_text());assert len(rows)==1653
for row in rows:
    p=pathlib.Path(row['raw_stageii_candidates'][0]);assert p.parent==root/'raw'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['raw_sha256']
available=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert available>8*1024**3
assert not (r/'receipts/local1653_launch.json').exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=['{PY}',str(r/'scripts/prepare_paper_data.py'),'--spec',str(r/'configs/data-prep-env-spec.json'),'--pairs',str(root/'pairs.private.json'),'--output',str(root/'prepared1653_v1')]
p=subprocess.Popen(cmd,env=env,stdout=open(r/'logs/data_prep1653.log','ab'),stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
receipt={{'pid':p.pid,'command':cmd,'verified_raw':len(rows),'source_sha256':hashlib.sha256((r/'scripts/prepare_paper_data.py').read_bytes()).hexdigest(),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'durability':'raw originals in localC and archivesD; RAM is working copy; outputs must be backed up to localD'}}
(r/'receipts/local1653_launch.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
    receipt=json.loads(remote(launch,timeout=180));state(status='RUNNING_GMR_CPU_ONLY',launch=receipt)
    last=-1
    for _ in range(2880):
        progress=json.loads(remote("import pathlib; p=pathlib.Path('"+RAM+"/prepared1653_v1/progress.json'); print(p.read_text() if p.exists() else '{}')",timeout=40))
        completed=progress.get('processed',0)
        if completed//100>last:state(status='RUNNING_GMR_CPU_ONLY',progress=progress);collect_publish();last=completed//100
        if progress.get('status') in ['PREPROCESSING_FINISHED_FILTER_PENDING','PREPROCESSING_FAILED']:break
        time.sleep(30)
    else:raise TimeoutError('Processing exceeded24h; server process and outputs preserved')
    state(status='BACKING_UP_PREPARED_PRIVATE_OUTPUT',progress=progress)
    target=LOCAL/'datasets/paper29_local1653_20261025.private.tar.gz';target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix(target.suffix+'.partial');assert not target.exists() and not partial.exists()
    with partial.open('xb') as output:
        proc=subprocess.run([*SSH,'tar -C '+shlex.quote(RAM)+' -czf - prepared1653_v1 pairs.private.json'],stdout=output,stderr=subprocess.PIPE,timeout=3600)
    assert proc.returncode==0,proc.stderr[-1000:]
    with tarfile.open(partial) as archive:
        members=archive.getmembers();assert sum(x.name.endswith('.private.npz') for x in members)==progress['completed_sequences']
        for member in members:
            if member.isfile():
                with archive.extractfile(member) as source:
                    while source.read(8*1024**2):pass
    digest=sha(partial);partial.rename(target)
    state(status='PREPROCESSING_BACKED_UP_FILTER_PENDING',progress=progress,archive_sha256=digest,archive_bytes=target.stat().st_size)
    collect_publish()

if __name__=='__main__':
    try:main()
    except BaseException as error:state(status='STOPPED_WITH_ERROR',error=repr(error));raise
