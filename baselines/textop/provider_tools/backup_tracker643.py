"""One private durable backup of the new complete FK references, never delete."""
import datetime,gzip,json,pathlib,shlex,shutil,subprocess,sys,tarfile,time
from session import REPO,SSH,remote
from restore_indexed_cache import sha
BASE=REPO/'baselines/textop';HERE=pathlib.Path(__file__).resolve().parent
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
STATE=BASE/'results/data_preparation_20261025/reference643_backup.json'
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
RAM='/dev/shm/ouyanghao_textop_20261025/tracker_references/export643_v1'
sys.stdout.reconfigure(encoding='utf-8')
def state(status,**values):
    old=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    old.update(status=status,time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**values)
    STATE.write_text(json.dumps(old,indent=2),encoding='utf-8');print(json.dumps(old),flush=True)
def main():
    assert not STATE.exists(),'Existing backup state retained; refuse duplicate'
    state('WAITING_FOR_COMPLETE_FK_EXPORT',raw_assets_published=False,deleted_files=0)
    for _ in range(2880):
        proof=json.loads(remote("import pathlib; p=pathlib.Path('"+ROOT+"/receipts/tracker_job_export643_v1_exit.json');print(p.read_text() if p.exists() else '{}')"))
        if proof:break
        time.sleep(30)
    else:raise TimeoutError('FK export not complete after24h')
    assert proof['child_exit_code']==0 and not proof['timed_out'] and proof['numerical_status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
    evidence=json.loads(remote(f'''
import pathlib,json,hashlib
r=pathlib.Path({RAM!r});receipt=json.loads((r/'receipt.json').read_text());assert receipt['exported_source_sequences']==643 and receipt['exported_trajectories']==1286
rows=[json.loads(x) for x in (r/'manifest.private.jsonl').read_text().splitlines()];assert len(rows)==1286
for row in rows:
 p=pathlib.Path(row['file']);assert p.parent==r and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
print(json.dumps({{'source_files_sha256_verified':len(rows),'private_manifest_sha256':receipt['private_manifest_sha256'],'source_bytes':sum(x.stat().st_size for x in r.iterdir() if x.is_file()),'physical_filter_executed':False}}))
''',timeout=180))
    state('VERIFIED_FK_REFERENCES_WAITING_FOR_BACKUP_SPACE',**evidence)
    target=LOCAL/'datasets/paper29_tracker_references643_20261025.private.tar.gz';partial=target.with_suffix(target.suffix+'.partial')
    assert not target.exists() and not partial.exists()
    for _ in range(2880):
        kit=BASE/'results/data_preparation_20261025/kit_resume.json'
        try:k=json.loads(kit.read_text(encoding='utf-8')) if kit.exists() else {}
        except json.JSONDecodeError:time.sleep(1);continue
        kit_remaining=max(0,k.get('expected_bytes',0)-k.get('bytes_received',0)) if k.get('status')!='LOCAL_ORIGINAL_ARCHIVE_VERIFIED' else 0
        required=int(evidence['source_bytes']*1.05)+kit_remaining+6*1024**3
        if shutil.disk_usage(LOCAL).free>required:break
        state('WAITING_FOR_DURABLE_BACKUP_SPACE',required_bytes=required);time.sleep(30)
    else:raise RuntimeError('Backup reserve unavailable; references and original inputs retained')
    state('BACKING_UP_PRIVATE_FK_REFERENCES',required_bytes=required)
    for attempt in range(3):
        with partial.open('xb') as output:
            proc=subprocess.run([*SSH,'tar --exclude=./usd -C '+shlex.quote(RAM)+' -czf - .'],stdout=output,stderr=subprocess.PIPE,timeout=7200)
        if proc.returncode==0:break
        partial.rename(partial.with_name(partial.name+'.failed_'+str(attempt+1)))
    assert proc.returncode==0
    with gzip.open(partial,'rb') as source:
        while source.read(8*1024**2):pass
    with tarfile.open(partial) as archive:assert sum(x.name.endswith('.private.npz') for x in archive.getmembers())==1286
    digest=sha(partial);partial.rename(target)
    state('PRIVATE_FK_REFERENCES_BACKED_UP',archive_sha256=digest,archive_bytes=target.stat().st_size,
          original_sources_retained=True,physical_filter_executed=False)
    for name in ['collect_data_preparation.py','collect_tracker_progress.py','publish.py']:
        p=subprocess.run([sys.executable,str(HERE/name)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        if p.returncode:raise RuntimeError(name+': '+p.stderr[-1200:])
if __name__=='__main__':
    try:main()
    except BaseException as error:state('BACKUP_STOPPED_ALL_DATA_RETAINED',error=repr(error));raise
