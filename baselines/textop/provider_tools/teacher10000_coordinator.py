"""Single bounded observer -> real backup -> exact-process evaluation handoff."""
import argparse,hashlib,json,msvcrt,os,pathlib,re,subprocess,sys,time,traceback
from session import remote
B=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path('D:/具身智能/TextOp复现_20261024');ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--once',action='store_true');a=p.parse_args();assert re.fullmatch('[a-z0-9_]+',a.run)
    guard=(D/'teacher10000_coordinator.lock').open('a+b')
    if guard.seek(0,2)==0:guard.write(b'0');guard.flush()
    guard.seek(0);msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
    folder=B/'results/teacher10000_evaluation_20261026'/a.run;folder.mkdir(exist_ok=False)
    state=dict(status='STARTING',pid=os.getpid(),evaluation_checkpoint=10000,signals_sent_by_coordinator=0,sdk_dispatches=0)
    sp=B/'configs/teacher10000-evaluation-spec.json';expected=hashlib.sha256(sp.read_bytes()).hexdigest()
    def persist():
        state['sample_unix']=time.time();tmp=folder/'state.partial';tmp.write_text(json.dumps(state,indent=2),encoding='utf-8');tmp.replace(folder/'state.json')
    def control(action):
        code='''import pathlib,hashlib,json,subprocess,sys
r=pathlib.Path(%r);sp=r/'configs/teacher10000-evaluation-spec.json';assert hashlib.sha256(sp.read_bytes()).hexdigest()==%r
v=subprocess.run([sys.executable,str(r/'scripts/teacher_evaluation_control.py'),'--action',%r],capture_output=True,text=True,timeout=90)
assert v.returncode==0,(v.stderr,v.stdout)
print(v.stdout)
'''%(ROOT,expected,action)
        return json.loads(remote(code,timeout=110))
    try:
        persist();deadline=time.monotonic()+48*3600
        while time.monotonic()<deadline:
            assert hashlib.sha256(sp.read_bytes()).hexdigest()==expected,'Live control config changed'
            obs=control('observe');state['observation']=obs;state['status']=obs['status'];persist()
            if a.once:state['status']='OBSERVE_ONLY_COMPLETED';persist();break
            if obs['status']=='USER_AUTHORIZED_TEACHER10000_EVALUATION_HANDOFF':break
            if obs['status']=='TEACHER10000_STOP_PENDING':
                if obs['all_original_exits_observed']:state.update(status='USER_AUTHORIZED_TEACHER10000_EVALUATION_HANDOFF',handoff=control('finish'));persist();break
                time.sleep(30);continue
            if not obs['target_checkpoint_ready']:time.sleep(60);continue
            state['status']='BACKING_UP_VERIFIED_10000';persist()
            v=subprocess.run([sys.executable,str(B/'provider_tools/backup_full_teacher_checkpoint.py'),'--completed','10000'],capture_output=True,text=True,encoding='utf-8',timeout=1800)
            (folder/'backup_execution.json').write_text(json.dumps(dict(exit_code=v.returncode,stdout=v.stdout,stderr=v.stderr),indent=2),encoding='utf-8');assert v.returncode==0,v.stderr
            proof=B/'results/full_teacher_checkpoints_20261026/round_10000/backup.json';raw=proof.read_text(encoding='utf-8')
            uploaded=remote('''import pathlib,json,hashlib
p=pathlib.Path(%r)/'receipts/teacher10000_backup_verified.json';raw=%r
if p.exists():assert p.read_text()==raw
else:
 with p.open('x') as f:f.write(raw)
print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
'''%(ROOT,raw),timeout=30)
            state.update(status='REQUESTING_BACKED_UP_EVALUATION_PAUSE',backup=json.loads(uploaded));persist()
            state.update(status='USER_AUTHORIZED_TEACHER10000_EVALUATION_HANDOFF',handoff=control('finish'));persist();break
        else:raise TimeoutError('48h observer bound reached; training unchanged')
    except BaseException:
        state.update(status='TEACHER10000_TRANSITION_NEEDS_INSPECTION',error=traceback.format_exc());persist();raise

if __name__=='__main__':main()
