"""Recover only the observer/backup worker, never restart current GMR.

GitHub sync is best effort and cannot terminate data processing or archival.
Private outputs stay local/server; only aggregate receipts are public.
"""
import datetime,gzip,hashlib,json,os,pathlib,shlex,shutil,subprocess,sys,tarfile,time
from session import REPO,SSH,remote
from restore_indexed_cache import sha
BASE=REPO/'baselines/textop';HERE=pathlib.Path(__file__).resolve().parent
TASK=REPO.parent.parent/'textop_reproduction_20261024';LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
DEST=BASE/'results/data_preparation_20261025';STATE=DEST/'local1653_queue.json'
RAM='/dev/shm/ouyanghao_textop_20261025/local1653_v1'
sys.stdout.reconfigure(encoding='utf-8');ENV=os.environ.copy();ENV['PYTHONIOENCODING']='utf-8'
def state(status,**values):
    old=json.loads(STATE.read_text(encoding='utf-8'));old.update(status=status,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**values)
    temp=TASK/'local1653_resume_state.tmp';temp.write_text(json.dumps(old,indent=2),encoding='utf-8');os.replace(temp,STATE)
    print(json.dumps({'status':status,**values}),flush=True)
def sync():
    failures=[]
    for name in ['collect_data_preparation.py','collect_storage_cleanup.py','collect_tracker_progress.py','publish.py']:
        try:
            r=subprocess.run([sys.executable,str(HERE/name)],env=ENV,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
            if r.returncode:failures.append({'script':name,'error':r.stderr[-1200:]})
        except Exception as error:failures.append({'script':name,'error_type':type(error).__name__})
    (DEST/'resume1653_sync_state.json').write_text(json.dumps({'status':'SYNC_PENDING' if failures else 'SYNC_VERIFIED',
        'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'failures':failures,
        'data_processing_and_backup_must_continue':True},indent=2),encoding='utf-8')
    return not failures
def progress():
    for attempt in range(6):
        try:return json.loads(remote("import pathlib; p=pathlib.Path('"+RAM+"/prepared1653_v1/progress.json');print(p.read_text())",timeout=40))
        except Exception:
            if attempt==5:raise
            time.sleep(min(2**attempt,16))
def main():
    ticket=DEST/'resume1653_watcher.json';assert not ticket.exists(),'Observer recovery already registered; refuse duplicate'
    old=json.loads(STATE.read_text(encoding='utf-8'));assert old['status']=='STOPPED_WITH_ERROR'
    shutil.copyfile(STATE,DEST/'local1653_queue_error_before_resume.json')
    ticket.write_text(json.dumps({'pid':os.getpid(),'scope':'observe existing GMR, then archive; no retarget/training launch',
        'recovering_after':'GitHub push failure stopped observer only','source_durable':'complete licensed originals on D; private recovery indexes; server RAM working copies'},indent=2),encoding='utf-8')
    j=progress();assert j['total_requested']==1653
    state('RUNNING_GMR_CPU_ONLY',progress=j,observer_recovered=True,source_durable='D complete original archives and verified private recovery kit; RAM is working copy')
    last=j.get('processed',0)//100;sync()
    for _ in range(5760):
        j=progress()
        if j.get('processed',0)//100>last:
            state('RUNNING_GMR_CPU_ONLY',progress=j);sync();last=j.get('processed',0)//100
        if j['status'] in ['PREPROCESSING_FINISHED_FILTER_PENDING','PREPROCESSING_FAILED']:break
        time.sleep(30)
    else:raise TimeoutError('Existing processing exceeded48h; all processes/outputs retained')
    state('BACKING_UP_PREPARED_PRIVATE_OUTPUT',progress=j)
    target=LOCAL/'datasets/paper29_local1653_20261025.private.tar.gz'
    assert not target.exists(),'Existing complete archive must be inspected rather than overwritten'
    partial=target.with_name(target.name+'.resume1.partial');assert not partial.exists()
    estimate=int(remote("import pathlib; r=pathlib.Path('"+RAM+"/prepared1653_v1');print(sum(x.stat().st_size for x in r.rglob('*') if x.is_file()))"))
    for _ in range(2880):
        if shutil.disk_usage(LOCAL).free>estimate*1.05+2*1024**3:break
        state('WAITING_FOR_DURABLE_BACKUP_SPACE',required_bytes=int(estimate*1.05+2*1024**3));time.sleep(30)
    else:raise RuntimeError('Private backup reserve unavailable; outputs retained')
    for attempt in range(3):
        with partial.open('xb') as output:
            r=subprocess.run([*SSH,'tar -C '+shlex.quote(RAM)+' -czf - prepared1653_v1 pairs.private.json'],stdout=output,stderr=subprocess.PIPE,timeout=7200)
        if r.returncode==0:break
        partial.rename(partial.with_name(partial.name+'.failed'+str(attempt+1)))
    assert r.returncode==0,'Backup transfer failed; partials and originals retained'
    digests={};expected=None
    with tarfile.open(partial,mode='r|gz') as archive:
        for member in archive:
            if not member.isfile():continue
            if member.name.endswith('prepared_manifest.private.jsonl'):
                expected={pathlib.Path(x['output_file']).name:x['output_sha256'] for x in map(json.loads,archive.extractfile(member).read().decode().splitlines())}
            elif member.name.endswith('.private.npz'):
                h=hashlib.sha256()
                with archive.extractfile(member) as source:
                    for b in iter(lambda:source.read(8388608),b''):h.update(b)
                digests[pathlib.PurePosixPath(member.name).name]=h.hexdigest()
    with gzip.open(partial,'rb') as source:
        for _ in iter(lambda:source.read(8388608),b''):pass
    assert expected is not None and digests==expected and len(digests)==j['completed_sequences']
    digest=sha(partial);partial.rename(target)
    state('PREPROCESSING_BACKED_UP_FILTER_PENDING' if j['completed_sequences']==1653 and j['failed_sequences']==0 else 'PREPROCESSING_PARTIAL_FAILURES_BACKED_UP',
          progress=j,archive_sha256=digest,archive_bytes=target.stat().st_size,per_motion_backup_sha256_verified=len(digests),observer_recovered=True)
    # CMU may now start from its existing waiter, independently of GitHub health.
    for _ in range(12):
        if sync():break
        time.sleep(30)
if __name__=='__main__':
    try:main()
    except BaseException as error:
        state('STOPPED_WITH_ERROR_DATA_AND_PROCESSES_RETAINED',error=repr(error));raise
