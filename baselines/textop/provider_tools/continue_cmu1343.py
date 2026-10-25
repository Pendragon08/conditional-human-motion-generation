"""Finite indexed CMU staging -> sequential GMR -> verified private backup.

The licensed complete archive remains on persistent C. Only selected raw inputs
are streamed into a private server RAM working directory. No raw C extraction,
GPU training, dataset deletion, or resumption of paused experiments occurs.
"""
import datetime,gzip,hashlib,json,os,pathlib,shlex,subprocess,sys,tarfile,time
from session import REPO,SSH,remote
from restore_indexed_cache import sha
sys.path.insert(0,str(REPO/'baselines/textop/scripts'))
from reconcile_raw_sources import source_key
BASE=REPO/'baselines/textop';HERE=pathlib.Path(__file__).resolve().parent
TASK=REPO.parent.parent/'textop_reproduction_20261024';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
SOURCE=TASK/'private_assets/incoming_amass/CMU_smplx_g.tar.bz2'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RAM='/dev/shm/ouyanghao_textop_20261025/cmu1330_v1'
PY='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'
STATE=BASE/'results/data_preparation_20261025/cmu1343_queue.json'
sys.stdout.reconfigure(encoding='utf-8')

def state(status,**values):
    old=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    old.update(status=status,time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**values)
    STATE.write_text(json.dumps(old,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in old.items() if k!='progress'}),flush=True)

def publish():
    for name in ['collect_data_preparation.py','collect_storage_cleanup.py','publish.py']:
        p=subprocess.run([sys.executable,str(HERE/name)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        if p.returncode:raise RuntimeError(name+': '+p.stderr[-1500:])

def main():
    assert not STATE.exists(),'Existing CMU continuation retained; refusing duplicate launch'
    evidence=json.loads((BASE/'results/data_preparation_20261025/cmu_resume.json').read_text(encoding='utf-8'))
    assert evidence['status']=='LOCAL_ORIGINAL_ARCHIVE_VERIFIED' and evidence['compressed_crc_and_tar_headers_verified']
    assert sha(SOURCE)==evidence['sha256']
    indexed=json.loads((LOCAL/'datasets/BABEL/textop_original_verified_pairs.private.json').read_text(encoding='utf-8'))
    wanted={source_key(x['source']):x for x in indexed if source_key(x['source']).split('/')[0]=='CMU'}
    assert len(wanted)==1343
    validation=json.loads((SOURCE.parent/'CMU_archive_validation.private.json').read_text(encoding='utf-8'))
    members={source_key(x['path']):x for x in validation['members'] if x['path'].endswith('_stageii.npz') and source_key(x['path']) in wanted}
    missing=sorted(set(wanted)-set(members))
    assert len(members)==1330 and len(missing)==13,'CMU coverage changed; inspect before staging'
    (LOCAL/'datasets/BABEL/cmu_missing_selected_export.private.json').write_text(json.dumps([wanted[k] for k in missing]),encoding='utf-8')
    wanted={k:v for k,v in wanted.items() if k in members}
    raw_bytes=sum(x['bytes'] for x in members.values())
    state('VERIFIED_ORIGINAL_ARCHIVE_PREPARING_INDEXED_STAGING',target=1330,requested_indexed_sources=1343,missing_from_selected_official_export=13,raw_bytes=raw_bytes,
          archive_sha256=evidence['sha256'],source_durable='complete C-drive original archive; private original-BABEL-verified index on D',
          ram_working_copy=True,physical_filter_executed=False)
    proof=json.loads(remote(f'''
import pathlib,os,json,shutil
assert os.getuid()==1013
r=pathlib.Path({RAM!r});assert not r.exists()
mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert mem>{raw_bytes}*1.5+8*1024**3
assert shutil.disk_usage('/dev/shm').free>{raw_bytes}*1.5+8*1024**3
assert r.parent.stat().st_uid==1013
r.mkdir(mode=0o700);(r/'raw').mkdir(mode=0o700)
print(json.dumps({{'available_memory_bytes':mem,'one_staging_cpu':sorted(os.sched_getaffinity(0))[0]}}))
'''))
    pairs=[];sent=0
    with (TASK/'cmu1343_transfer_stderr.log').open('xb') as err:
        command='taskset -c '+str(proof['one_staging_cpu'])+' tar -xzf - -C '+shlex.quote(RAM+'/raw')
        child=subprocess.Popen([*SSH,command],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=err)
        try:
            with tarfile.open(SOURCE,mode='r|bz2') as original,tarfile.open(fileobj=child.stdin,mode='w|gz',compresslevel=1) as outgoing:
                for member in original:
                    if not member.isfile() or not member.name.endswith('_stageii.npz'):continue
                    key=source_key(member.name)
                    if key not in wanted:continue
                    name=hashlib.sha256(key.encode()).hexdigest()[:20]+'_stageii.npz'
                    digest=hashlib.sha256()
                    class HashedReader:
                        def __init__(self,source):self.source=source
                        def read(self,size=-1):
                            block=self.source.read(size);digest.update(block);return block
                    info=tarfile.TarInfo(name);info.size=member.size;info.mode=0o600
                    with original.extractfile(member) as source:outgoing.addfile(info,HashedReader(source))
                    pairs.append({**wanted[key],'raw_stageii_candidates':[RAM+'/raw/'+name],'raw_sha256':digest.hexdigest(),
                                  'archive_member':member.name,'archive_sha256':evidence['sha256']})
                    sent+=member.size
                    if len(pairs)%100==0:state('STAGING_SELECTED_RAW_IN_SERVER_RAM',staged_files=len(pairs),staged_bytes=sent)
            child.stdin.close();assert child.wait(timeout=180)==0
        except BaseException:
            child.stdin.close();child.wait(timeout=180);raise
    assert len(pairs)==1330 and sent==raw_bytes
    manifest=LOCAL/'datasets/BABEL/cmu1343_runtime_pairs.private.json'
    manifest.write_text(json.dumps(pairs),encoding='utf-8')
    subprocess.run(['scp','-q','-o','BatchMode=yes','-i','C:/Users/Pende/.ssh/id_ed25519',str(manifest),'ouyanghao@10.147.17.32:'+RAM+'/pairs.private.json'],check=True)
    verified=json.loads(remote(f'''
import pathlib,json,hashlib,os
r=pathlib.Path({RAM!r});rows=json.loads((r/'pairs.private.json').read_text());assert len(rows)==1330
for row in rows:
 p=pathlib.Path(row['raw_stageii_candidates'][0]);assert p.parent==r/'raw' and p.stat().st_uid==1013
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['raw_sha256']
print(json.dumps({{'files_sha256_verified':len(rows),'raw_bytes':sum(pathlib.Path(x['raw_stageii_candidates'][0]).stat().st_size for x in rows)}}))
''',timeout=180))
    state('STAGED_VERIFIED_WAITING_FOR_PREVIOUS_BATCH_BACKUP',verification=verified)
    previous=BASE/'results/data_preparation_20261025/local1653_queue.json'
    for _ in range(2880):
        try:prior=json.loads(previous.read_text(encoding='utf-8'))
        except json.JSONDecodeError:time.sleep(1);continue
        if prior.get('status')=='PREPROCESSING_BACKED_UP_FILTER_PENDING':break
        time.sleep(30)
    else:raise TimeoutError('Previous1653 preprocessing/backup not complete after24h; CMU working data retained')
    source_hash=sha(BASE/'scripts/prepare_paper_data.py')
    launch=json.loads(remote(f'''
import pathlib,hashlib,subprocess,os,json,time
r=pathlib.Path({ROOT!r});ram=pathlib.Path({RAM!r})
assert not (r/'receipts/cmu1343_launch.json').exists()
assert hashlib.sha256((r/'scripts/prepare_paper_data.py').read_bytes()).hexdigest()=={source_hash!r}
old=pathlib.Path('/proc/3319567/status')
assert not old.exists() or next(x for x in old.read_text().splitlines() if x.startswith('State:')).split()[1]=='Z'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[{PY!r},str(r/'scripts/prepare_paper_data.py'),'--spec',str(r/'configs/data-prep-env-spec.json'),'--pairs',str(ram/'pairs.private.json'),'--output',str(ram/'prepared1330_v1')]
p=subprocess.Popen(cmd,env=env,stdout=open(r/'logs/data_prep_cmu1343.log','ab'),stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
proof={{'pid':p.pid,'command':cmd,'cpu_only':True,'source_sha256':{source_hash!r},'target':1330,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}}
(r/'receipts/cmu1343_launch.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''))
    state('RUNNING_GMR_CPU_ONLY',launch=launch);last=-1
    for _ in range(5760):
        try:progress=json.loads(remote("import pathlib; p=pathlib.Path('"+RAM+"/prepared1330_v1/progress.json');print(p.read_text() if p.exists() else '{}')",timeout=40))
        except json.JSONDecodeError:time.sleep(1);continue
        if progress.get('processed',0)//100>last:
            state('RUNNING_GMR_CPU_ONLY',progress=progress);publish();last=progress.get('processed',0)//100
        if progress.get('status') in ['PREPROCESSING_FINISHED_FILTER_PENDING','PREPROCESSING_FAILED']:break
        time.sleep(30)
    else:raise TimeoutError('CMU processing exceeded48h; all working outputs retained')
    state('BACKING_UP_PRIVATE_PREPARED_OUTPUT',progress=progress)
    target=LOCAL/'datasets/paper29_cmu1330_20261025.private.tar.gz';partial=target.with_suffix(target.suffix+'.partial')
    assert not target.exists() and not partial.exists()
    import shutil
    estimate=int(remote("import pathlib; p=pathlib.Path('"+RAM+"/prepared1330_v1');print(sum(x.stat().st_size for x in p.rglob('*') if x.is_file()))"))
    for _ in range(2880):
        if shutil.disk_usage(LOCAL).free>estimate*1.05+2*1024**3:break
        state('WAITING_FOR_DURABLE_BACKUP_SPACE',required_bytes=int(estimate*1.05+2*1024**3));time.sleep(30)
    else:raise RuntimeError('Insufficient backup space; all RAM outputs and original archives retained')
    for attempt in range(3):
        with partial.open('xb') as output:
            p=subprocess.run([*SSH,'tar -C '+shlex.quote(RAM)+' -czf - prepared1330_v1 pairs.private.json'],stdout=output,stderr=subprocess.PIPE,timeout=7200)
        if p.returncode==0:break
        partial.rename(partial.with_name(partial.name+'.failed_'+str(attempt+1)))
    assert p.returncode==0,'All backup attempts failed; working outputs and failures retained'
    with gzip.open(partial,'rb') as f:
        while f.read(8*1024**2):pass
    with tarfile.open(partial) as archive:
        members=archive.getmembers();assert sum(x.name.endswith('.private.npz') for x in members)==progress['completed_sequences']
    digest=sha(partial);partial.rename(target)
    state('PREPROCESSING_BACKED_UP_FILTER_PENDING',progress=progress,archive_sha256=digest,archive_bytes=target.stat().st_size)
    publish()

if __name__=='__main__':
    try:main()
    except BaseException as error:state('STOPPED_WITH_ERROR_ALL_DATA_RETAINED',error=repr(error));raise
