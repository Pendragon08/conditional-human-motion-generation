"""Bind the requested eval, SuperMimic, video chain after existing-sample validation."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

S=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029/repair_prior29_v2_20261105')
HERE=Path(__file__).resolve().parent
OUT=S/'paper_pipeline_v7_launch'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)

def prepare():
    receipt=OUT/'followup_waiter_launch.json'
    if receipt.exists():print(receipt.read_text());return
    assert (S/'paper_protocol_validation_v7/launch.json').exists()
    legacy=S/'src/supermimic_paper_legacy_v7'
    legacy.mkdir(exist_ok=False)
    for src in (S/'src/supermimic_video_legacy_v4').glob('*.py'):
        shutil.copyfile(src,legacy/src.name)
        assert sha(src)==sha(legacy/src.name)
    shutil.copyfile(HERE/'evaluate_native2000_tasks_paper_v7.py',legacy/'evaluate_native2000_tasks_v2_20261105.py')
    for name in ['evaluation_controller_v7.py','supermimic_repaired_controller_v7.py','report_video_sources_v4.py','tokenize_paper_pour_v7.py']:
        target=S/'src'/name
        assert not target.exists()
        shutil.copyfile(HERE/name,target)
    sources=[*legacy.glob('*.py'),*(S/'src'/n for n in ['evaluation_controller_v7.py','supermimic_repaired_controller_v7.py','report_video_sources_v4.py','tokenize_paper_pour_v7.py'])]
    for p in sources:compile(p.read_text(),str(p),'exec')
    write(OUT/'followup_sources.json',{str(p):sha(p) for p in sources})
    with (OUT/'followup_waiter.log').open('xb') as log:
        child=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker'],
             cwd=S,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(receipt,{'pid':child.pid,'source_sha256':sha(__file__),'source_binding_sha256':sha(OUT/'followup_sources.json')})
    print(receipt.read_text())

def work():
    import fcntl
    with (OUT/'followup_waiter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=read(S/'paper_protocol_validation_v7/status.json')
            if state['status'] in ['failed_preserved','existing_validation_rejected','existing_native_rejected']:
                write(OUT/'followups_status.json',{'status':'prerequisite_rejected_preserved','validation':state['status']});return
            if state['status']=='existing_sample_validation_passed' and (OUT/'launch.json').exists():break
            write(OUT/'followups_status.json',{'status':'waiting_existing_sample_validation'})
            time.sleep(15)
        for p,h in read(OUT/'followup_sources.json').items():assert sha(p)==h
        plan=read(OUT/'config.json')
        launched={}
        for key,name in [('tokens','tokenize_paper_pour_v7.py'),('evaluation','evaluation_controller_v7.py'),
                         ('supermimic','supermimic_repaired_controller_v7.py'),('videos','report_video_sources_v4.py')]:
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
            if key=='videos':env.update(plan['postopt_env']);env['JAX_PLATFORMS']='cpu'
            env['PYTHONPATH']=str(S/'src')+os.pathsep+env.get('PYTHONPATH','')
            with (OUT/(key+'.log')).open('xb') as f:
                child=subprocess.Popen([plan['sample_python'],'-B','-u',str(S/'src'/name)],
                  cwd=S,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            launched[key]={'pid':child.pid,'source':str(S/'src'/name),'source_sha256':sha(S/'src'/name)}
            write(OUT/'followups_launch.json',launched)
        write(OUT/'followups_status.json',{'status':'followups_started_waiting_verified_data','processes':launched})

if __name__=='__main__':work() if sys.argv[1:]==['--worker'] else prepare()
