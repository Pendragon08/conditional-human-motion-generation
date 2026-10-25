"""Finite official CMU resume, using a fresh in-memory authorized AMASS login.

Credentials are entered hidden and sent only over an anonymous child stdin pipe.
Old D-drive partial is retained. No browser cookie/profile access, no raw Git upload.
"""
import argparse,datetime,getpass,hashlib,http.cookiejar,json,os,pathlib,shutil,subprocess,sys,time,urllib.parse,urllib.request
from session import REPO
from restore_indexed_cache import sha
sys.stdout.reconfigure(encoding='utf-8')
BASE=REPO/'baselines/textop';TASK=REPO.parent.parent/'textop_reproduction_20261024'
LOCAL=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ROOT=TASK/'private_assets/incoming_amass';ROOT.mkdir(parents=True,exist_ok=True)
OLD=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives/CMU_smplx_g.tar.bz2.part')
FINAL=ROOT/'CMU_smplx_g.tar.bz2';PART=ROOT/'CMU_smplx_g.tar.bz2.part'
URL='https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/gender_specific/mosh_results/CMU.tar.bz2'
EXPECTED=11572134418
STATE=BASE/'results/data_preparation_20261025/cmu_resume.json'

def state(status,**kw):
    value={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'status':status,'subset':'CMU','official_url':URL,'expected_bytes':EXPECTED,
           'credentials_stored':False,'old_D_partial_retained':True,'raw_data_published':False,**kw}
    STATE.write_text(json.dumps(value,indent=2),encoding='utf-8')
    (LOCAL/'CMU续传状态_20261025.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value),flush=True)

def worker():
    account=json.load(sys.stdin)
    jar=http.cookiejar.CookieJar()
    client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar),urllib.request.ProxyHandler({'https':'http://127.0.0.1:7897'}))
    client.addheaders=[('User-Agent','AMASS authorized research asset acquisition')]
    with client.open(URL,timeout=60) as response:response.read(4096)
    form=urllib.parse.urlencode({**account,'commit':'Log in'}).encode()
    with client.open(urllib.request.Request(URL,data=form),timeout=60) as response:
        assert response.read(4).startswith(b'BZh'),'Official file host did not authorize the download; no password retry'
    del form,account
    with client.open(urllib.request.Request(URL,method='HEAD'),timeout=60) as response:
        assert 'text/html' not in response.headers.get('Content-Type','')
        assert int(response.headers['Content-Length'])==EXPECTED,'Remote archive size changed'
    state('AUTHENTICATED_SIZE_VERIFIED',old_partial_bytes=OLD.stat().st_size)
    assert not FINAL.exists()
    if not PART.exists():
        assert shutil.disk_usage(ROOT).free>EXPECTED+6*1024**3,'6GiB system reserve required'
        old_hash=sha(OLD);shutil.copyfile(OLD,PART);assert sha(PART)==old_hash
        state('D_PARTIAL_COPIED_AND_SHA_VERIFIED',copied_bytes=PART.stat().st_size,partial_sha256=old_hash)
    assert PART.stat().st_size<=EXPECTED
    for attempt in range(3):
        offset=PART.stat().st_size
        if offset==EXPECTED:break
        assert shutil.disk_usage(ROOT).free>EXPECTED-offset+6*1024**3
        # Verify a boundary from the official server before appending any bytes.
        start=max(0,offset-65536)
        with client.open(urllib.request.Request(URL,headers={'Range':f'bytes={start}-{offset-1}'}),timeout=90) as response:
            assert response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {start}-')
            remote_tail=response.read(offset-start)
        with PART.open('rb') as source:source.seek(start);assert source.read(offset-start)==remote_tail
        try:
            with client.open(urllib.request.Request(URL,headers={'Range':f'bytes={offset}-'}),timeout=90) as response,PART.open('ab') as output:
                assert response.status==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                assert 'text/html' not in response.headers.get('Content-Type','')
                last=0
                while True:
                    block=response.read(4*1024**2)
                    if not block:break
                    assert shutil.disk_usage(ROOT).free>len(block)+6*1024**3,'Space reserve reached; partial retained'
                    output.write(block)
                    if time.monotonic()-last>15:
                        output.flush();state('DOWNLOADING',bytes_received=PART.stat().st_size,attempt=attempt+1);last=time.monotonic()
            assert PART.stat().st_size==EXPECTED,'Partial response retained for retry'
        except (TimeoutError,OSError) as error:
            state('NETWORK_RETRY_PARTIAL_RETAINED',error_type=type(error).__name__,bytes_received=PART.stat().st_size)
            if attempt==2:raise
    assert PART.stat().st_size==EXPECTED
    state('VALIDATING_FULL_BZIP2_CRC_AND_TAR_HEADERS',bytes_received=PART.stat().st_size)
    sys.path.insert(0,str(BASE.parent/'phygile/scripts'))
    from download_amass_archives import validate_archive
    verification=validate_archive(PART)
    (ROOT/'CMU_archive_validation.private.json').write_text(json.dumps(verification),encoding='utf-8')
    PART.rename(FINAL)
    public={k:v for k,v in verification.items() if k!='members'}
    state('LOCAL_ORIGINAL_ARCHIVE_VERIFIED',**public,indexed_retargeting_executed=False,physical_filter_executed=False)

def launch():
    username=getpass.getpass('Official AMASS account (hidden): ')
    password=getpass.getpass('Official AMASS password (hidden): ')
    log=TASK/'cmu_resume_stdout.log';error=TASK/'cmu_resume_stderr.log'
    with log.open('xb') as out,error.open('xb') as err:
        child=subprocess.Popen([sys.executable,'-u',str(pathlib.Path(__file__).resolve()),'--worker'],stdin=subprocess.PIPE,stdout=out,stderr=err,
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
        child.stdin.write(json.dumps({'username':username,'password':password}).encode());child.stdin.close()
    del username,password
    (TASK/'cmu_resume_launch.json').write_text(json.dumps({'pid':child.pid,'credential_transport':'anonymous stdin pipe only','credentials_stored':False}),encoding='utf-8')
    print(json.dumps({'pid':child.pid,'status':'AUTHENTICATION_PENDING'}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:
        try:worker()
        except BaseException as e:state('STOPPED_WITH_ERROR_PARTIAL_RETAINED',error_type=type(e).__name__,message=str(e)[:200]);raise
    else:launch()
