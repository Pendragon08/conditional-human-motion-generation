"""Finite BMLrub/KIT indexed preprocessing after download and prior backup.

No training or deletion. Full original archives remain durable. Selected raw
files are streamed to personal RAM with full SHA verification; GMR is serial.
Absent/ambiguous exported motions are recorded privately, never fabricated.
"""
import argparse,collections,datetime,gzip,hashlib,json,os,pathlib,shlex,shutil,subprocess,sys,tarfile,time
from session import REPO,SSH,remote
from restore_indexed_cache import sha
sys.path.insert(0,str(REPO/'baselines/textop/scripts'))
from reconcile_raw_sources import source_key
BASE=REPO/'baselines/textop';HERE=pathlib.Path(__file__).resolve().parent
TASK=REPO.parent.parent/'textop_reproduction_20261024';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
PY='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'
sys.stdout.reconfigure(encoding='utf-8')

def load(path):
    for _ in range(10):
        try:return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        except json.JSONDecodeError:time.sleep(1)
    raise RuntimeError('Repeated incomplete state: '+str(path))

def main(subset):
    tag=subset.lower();dest=BASE/'results/data_preparation_20261025';state_path=dest/(tag+'_indexed_queue.json')
    assert not state_path.exists(),'Existing continuation retained; inspect before retry'
    def state(status,**values):
        old=load(state_path);old.update(status=status,subset=subset,time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**values)
        state_path.write_text(json.dumps(old,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in old.items() if k!='progress'}),flush=True)
    def publish():
        for name in ['record_asset_authorization.py','collect_data_preparation.py','collect_storage_cleanup.py','publish.py']:
            result=subprocess.run([sys.executable,str(HERE/name)],capture_output=True,text=True,encoding='utf-8',errors='replace')
            if result.returncode:raise RuntimeError(name+': '+result.stderr[-1200:])
    try:
        state('WAITING_FOR_VERIFIED_DOWNLOAD_AND_PREVIOUS_GMR_BACKUP',physical_filter_executed=False,
              dataset_accepted=False,full_training_started=False)
        predecessor=dest/('cmu1343_queue.json' if subset=='BMLrub' else 'bmlrub_indexed_queue.json')
        for _ in range(8640):
            evidence=load(dest/(tag+'_resume.json'));previous=load(predecessor)
            if evidence.get('status')=='LOCAL_ORIGINAL_ARCHIVE_VERIFIED' and previous.get('status')=='PREPROCESSING_BACKED_UP_FILTER_PENDING':break
            if evidence.get('status')=='STOPPED_WITH_ERROR_PARTIAL_RETAINED':raise RuntimeError('Download failed; no new preprocessing launched')
            if previous.get('status','').startswith('STOPPED_WITH_ERROR'):raise RuntimeError('Predecessor failed; its data retained and no overlapping GMR launched')
            time.sleep(30)
        else:raise TimeoutError('Prerequisites not complete in72h; all data retained')
        source=(TASK/'private_assets/incoming_amass' if subset=='BMLrub' else pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives'))/(subset+'_smplx_g.tar.bz2')
        assert evidence['compressed_crc_and_tar_headers_verified'] and sha(source)==evidence['sha256']
        indexed=load(LOCAL/'datasets/BABEL/textop_original_verified_pairs.private.json')
        wanted={source_key(x['source']):x for x in indexed if source_key(x['source']).split('/')[0]==subset}
        assert len(wanted)==(2139 if subset=='BMLrub' else 2513)
        inventory=load(TASK/'private_assets/incoming_amass'/(subset+'_archive_validation.private.json'))
        matches=collections.defaultdict(list)
        for row in inventory['members']:
            if row['path'].endswith('_stageii.npz') and source_key(row['path']) in wanted:matches[source_key(row['path'])].append(row)
        assert not any(len(x)>1 for x in matches.values()),'Ambiguous exact archive key; inspect before staging'
        members={k:v[0] for k,v in matches.items()};missing=sorted(set(wanted)-set(members))
        (LOCAL/'datasets/BABEL'/(tag+'_missing_selected_export.private.json')).write_text(json.dumps([wanted[k] for k in missing]),encoding='utf-8')
        selected={k:wanted[k] for k in members};assert selected
        raw_bytes=sum(x['bytes'] for x in members.values());ram='/dev/shm/ouyanghao_textop_20261025/'+tag+'_indexed_v1'
        state('EXACT_SOURCE_RECONCILIATION_COMPLETE',requested_indexed_sources=len(wanted),target=len(selected),
              missing_from_selected_official_export=len(missing),raw_bytes=raw_bytes,archive_sha256=evidence['sha256'],
              source_durable='complete original archive on personal persistent volume; verified BABEL index on D',ram_working_copy=True)
        for _ in range(8640):
            available=json.loads(remote("import pathlib,json,shutil; print(json.dumps({'ram_free':shutil.disk_usage('/dev/shm').free,'mem_available':int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024}))"))
            if min(available.values())>raw_bytes*1.5+8*1024**3:break
            state('WAITING_FOR_SHARED_MEMORY_RESERVE',required_free_bytes=int(raw_bytes*1.5+8*1024**3),observed=available);time.sleep(30)
        else:raise TimeoutError('RAM reserve unavailable; persistent originals retained')
        cpu=int(remote(f"import pathlib,os; r=pathlib.Path({ram!r});assert os.getuid()==1013 and r.parent.stat().st_uid==1013 and not r.exists();r.mkdir(mode=0o700);(r/'raw').mkdir(mode=0o700);print(sorted(os.sched_getaffinity(0))[0])"))
        pairs=[];sent=0
        with (TASK/(tag+'_indexed_transfer_stderr.log')).open('xb') as error:
            child=subprocess.Popen([*SSH,'taskset -c '+str(cpu)+' tar -xzf - -C '+shlex.quote(ram+'/raw')],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=error)
            try:
                with tarfile.open(source,mode='r|bz2') as original,tarfile.open(fileobj=child.stdin,mode='w|gz',compresslevel=1) as outgoing:
                    for member in original:
                        if not member.isfile() or not member.name.endswith('_stageii.npz'):continue
                        key=source_key(member.name)
                        if key not in selected:continue
                        assert member.name==members[key]['path'] and member.size==members[key]['bytes']
                        name=hashlib.sha256(key.encode()).hexdigest()[:20]+'_stageii.npz';digest=hashlib.sha256()
                        class HashedReader:
                            def __init__(self,source):self.source=source
                            def read(self,size=-1):
                                block=self.source.read(size);digest.update(block);return block
                        info=tarfile.TarInfo(name);info.size=member.size;info.mode=0o600
                        with original.extractfile(member) as stream:outgoing.addfile(info,HashedReader(stream))
                        pairs.append({**selected[key],'raw_stageii_candidates':[ram+'/raw/'+name],
                                      'raw_sha256':digest.hexdigest(),'archive_member':member.name,'archive_sha256':evidence['sha256']});sent+=member.size
                        if len(pairs)%100==0:state('STAGING_SELECTED_RAW_IN_SERVER_RAM',staged_files=len(pairs),staged_bytes=sent)
                child.stdin.close();assert child.wait(timeout=180)==0
            except BaseException:
                child.stdin.close();child.wait(timeout=180);raise
        assert len(pairs)==len(selected) and sent==raw_bytes
        manifest=LOCAL/'datasets/BABEL'/(tag+'_runtime_pairs.private.json');manifest.write_text(json.dumps(pairs),encoding='utf-8')
        subprocess.run(['scp','-q','-o','BatchMode=yes','-i','C:/Users/Pende/.ssh/id_ed25519',str(manifest),'ouyanghao@10.147.17.32:'+ram+'/pairs.private.json'],check=True)
        source_hash=sha(BASE/'scripts/prepare_paper_data.py')
        launch=json.loads(remote(f'''
import pathlib,json,hashlib,subprocess,os,time
r=pathlib.Path({ROOT!r});ram=pathlib.Path({ram!r});rows=json.loads((ram/'pairs.private.json').read_text())
assert len(rows)=={len(selected)} and os.getuid()==1013
for row in rows:
 p=pathlib.Path(row['raw_stageii_candidates'][0]);assert p.parent==ram/'raw' and p.stat().st_uid==1013
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['raw_sha256']
assert hashlib.sha256((r/'scripts/prepare_paper_data.py').read_bytes()).hexdigest()=={source_hash!r}
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode();owner=p.stat().st_uid
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 assert not(owner==1013 and 'scripts/prepare_paper_data.py' in cmd and '--output' in cmd),'Another GMR is active'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[{PY!r},str(r/'scripts/prepare_paper_data.py'),'--spec',str(r/'configs/data-prep-env-spec.json'),'--pairs',str(ram/'pairs.private.json'),'--output',str(ram/'prepared_v1')]
with (r/'logs/data_prep_{tag}_indexed.log').open('xb') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
proof={{'pid':p.pid,'command':cmd,'cpu_only':True,'verified_raw':len(rows),'source_sha256':{source_hash!r}}}
(r/'receipts/{tag}_indexed_launch.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
''',timeout=180))
        state('RUNNING_GMR_CPU_ONLY',launch=launch);last=-1
        for _ in range(8640):
            progress=json.loads(remote("import pathlib;p=pathlib.Path('"+ram+"/prepared_v1/progress.json');print(p.read_text() if p.exists() else '{}')"))
            if progress.get('processed',0)//100>last:
                state('RUNNING_GMR_CPU_ONLY',progress=progress);publish();last=progress.get('processed',0)//100
            if progress.get('status') in ['PREPROCESSING_FINISHED_FILTER_PENDING','PREPROCESSING_FAILED']:break
            time.sleep(30)
        else:raise TimeoutError('GMR exceeded72h; server process and all outputs retained')
        estimate=int(remote("import pathlib;p=pathlib.Path('"+ram+"/prepared_v1');print(sum(x.stat().st_size for x in p.rglob('*') if x.is_file()))"))
        for _ in range(8640):
            if shutil.disk_usage(LOCAL).free>estimate*1.05+2*1024**3:break
            state('WAITING_FOR_DURABLE_BACKUP_SPACE',required_bytes=int(estimate*1.05+2*1024**3),progress=progress);time.sleep(30)
        else:raise RuntimeError('Insufficient durable backup space; outputs and originals retained')
        state('BACKING_UP_PRIVATE_PREPARED_OUTPUT',progress=progress)
        target=LOCAL/'datasets'/('paper29_'+tag+'_20261025.private.tar.gz');partial=target.with_suffix(target.suffix+'.partial')
        assert not target.exists() and not partial.exists()
        for attempt in range(3):
            with partial.open('xb') as output:proc=subprocess.run([*SSH,'tar -C '+shlex.quote(ram)+' -czf - prepared_v1 pairs.private.json'],stdout=output,stderr=subprocess.PIPE,timeout=7200)
            if proc.returncode==0:break
            partial.rename(partial.with_name(partial.name+'.failed_'+str(attempt+1)))
        assert proc.returncode==0
        with gzip.open(partial,'rb') as f:
            while f.read(8*1024**2):pass
        with tarfile.open(partial) as archive:assert sum(x.name.endswith('.private.npz') for x in archive.getmembers())==progress['completed_sequences']
        digest=sha(partial);partial.rename(target)
        state('PREPROCESSING_BACKED_UP_FILTER_PENDING' if progress['failed_sequences']==0 else 'PREPROCESSING_PARTIAL_FAILURES_BACKED_UP',
              progress=progress,archive_sha256=digest,archive_bytes=target.stat().st_size);publish()
    except BaseException as error:state('STOPPED_WITH_ERROR_ALL_DATA_RETAINED',error=repr(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subset',required=True,choices=['BMLrub','KIT']);main(p.parse_args().subset)
