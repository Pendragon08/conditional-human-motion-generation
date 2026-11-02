"""Two bounded official AMASS resumes using the same verified transfer logic.

BMLrub is completed on C and KIT on D; both retain6GiB volume reserve. Existing
partials remain intact, credentials use hidden entry/anonymous pipes only.
"""
import argparse,datetime,getpass,json,os,pathlib,subprocess,sys,time
from session import REPO
import resume_cmu_asset as transfer
BASE=REPO/'baselines/textop';TASK=REPO.parent.parent/'textop_reproduction_20261024'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ADIR=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives')
ITEMS={'BMLrub':12919496284,'KIT':13182414052}
sys.stdout.reconfigure(encoding='utf-8')

def worker(subset):
    filename=subset+'_smplx_g.tar.bz2';dest=TASK/'private_assets/incoming_amass' if subset=='BMLrub' else ADIR
    transfer.ROOT=dest;transfer.OLD=ADIR/(filename+'.part');transfer.FINAL=dest/filename;transfer.PART=dest/(filename+'.part')
    transfer.URL='https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/gender_specific/mosh_results/'+subset+'.tar.bz2'
    transfer.EXPECTED=ITEMS[subset];transfer.VALIDATION=TASK/'private_assets/incoming_amass'/(subset+'_archive_validation.private.json')
    def state(status,**values):
        obj={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             'status':status,'subset':subset,'official_url':transfer.URL,'expected_bytes':ITEMS[subset],
             'credentials_stored':False,'old_partial_retained_until_full_validation':True,
             'raw_data_published':False,'volume_reserve_bytes':6*1024**3,**values}
        (BASE/'results/data_preparation_20261025'/(subset.lower()+'_resume.json')).write_text(json.dumps(obj,indent=2),encoding='utf-8')
        (LOCAL/(subset+'续传状态_20261025.json')).write_text(json.dumps(obj,indent=2),encoding='utf-8');print(json.dumps(obj),flush=True)
    transfer.state=state
    try:transfer.worker()
    except BaseException as error:state('STOPPED_WITH_ERROR_PARTIAL_RETAINED',error_type=type(error).__name__,message=str(error)[:200]);raise

def watch():
    done=set()
    for _ in range(2880):
        for subset in ITEMS:
            p=BASE/'results/data_preparation_20261025'/(subset.lower()+'_resume.json')
            try:obj=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
            except json.JSONDecodeError:continue
            if subset not in done and obj.get('status') in ['LOCAL_ORIGINAL_ARCHIVE_VERIFIED','STOPPED_WITH_ERROR_PARTIAL_RETAINED']:
                for name in ['record_asset_authorization.py','collect_data_preparation.py','collect_storage_cleanup.py','publish.py']:
                    proc=subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name(name))],capture_output=True,text=True,encoding='utf-8',errors='replace')
                    if proc.returncode:raise RuntimeError(name+': '+proc.stderr[-1600:])
                done.add(subset);print(subset+' terminal result synchronized',flush=True)
        if len(done)==len(ITEMS):return
        time.sleep(30)
    raise TimeoutError('24h report continuation ended; all transfer workers/data retained')

def launch():
    assert not (TASK/'remaining_assets_launch.json').exists(),'Existing launch retained; inspect before retry'
    username=getpass.getpass('Official AMASS account (hidden): ');password=getpass.getpass('Official AMASS password (hidden): ')
    jobs=[]
    for subset in ITEMS:
        with (TASK/(subset.lower()+'_resume_stdout.log')).open('xb') as out,(TASK/(subset.lower()+'_resume_stderr.log')).open('xb') as error:
            child=subprocess.Popen([sys.executable,'-u',str(pathlib.Path(__file__).resolve()),'--worker',subset],stdin=subprocess.PIPE,stdout=out,stderr=error,
                                   creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
            child.stdin.write(json.dumps({'username':username,'password':password}).encode());child.stdin.close()
            jobs.append({'subset':subset,'pid':child.pid})
    del username,password
    with (TASK/'remaining_assets_watch_stdout.log').open('xb') as out,(TASK/'remaining_assets_watch_stderr.log').open('xb') as error:
        p=subprocess.Popen([sys.executable,'-u',str(pathlib.Path(__file__).resolve()),'--watch'],stdout=out,stderr=error,
                           creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    proof={'jobs':jobs,'terminal_report_worker_pid':p.pid,'credentials_stored':False,'credential_transport':'anonymous stdin pipes only'}
    (TASK/'remaining_assets_launch.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=list(ITEMS));p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.worker:worker(a.worker)
    elif a.watch:watch()
    else:launch()
