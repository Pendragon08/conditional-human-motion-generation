"""Run only the additional143 existing raw inputs after the active500 batch.

Serial CPU continuation, no RL or generator launch, no deletion or resumption.
"""
import fcntl, hashlib, json, os, pathlib, subprocess, time
R=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
D=R/'data_preparation_20261025'
PY='/data/code/ouyanghao/oyh/phygile-reproduction-20261017/envs/gmr_amass_20261021/bin/python'

def main():
    lock=(R/'receipts/additional143.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=R/'receipts/additional143_launch.json'
    if receipt.exists():raise FileExistsError('Previous continuation launch preserved')
    for _ in range(720):
        progress=json.loads((D/'available500_v1/progress.json').read_text())
        if progress['status']!='RUNNING':break
        time.sleep(30)
    else:raise TimeoutError('Active batch has not finished; no duplicate processing launched')
    assert progress['completed_sequences']==500 and progress['failed_sequences']==0,progress
    pairs=D/'additional_existing_raw_pairs.private.json'
    assert len(json.loads(pairs.read_text()))==143
    out=D/'available143_v1'
    if out.exists():raise FileExistsError('Existing output preserved')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    cmd=[PY,str(R/'scripts/prepare_paper_data.py'),'--spec',str(R/'configs/data-prep-env-spec.json'),'--pairs',str(pairs),'--output',str(out)]
    proc=subprocess.Popen(cmd,env=env,stdout=open(R/'logs/data_prep143.log','ab'),stderr=subprocess.STDOUT,preexec_fn=lambda:os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]))
    info={'pid':proc.pid,'command':cmd,'source_sha256':hashlib.sha256((R/'scripts/prepare_paper_data.py').read_bytes()).hexdigest(),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    receipt.write_text(json.dumps(info,indent=2));info['returncode']=proc.wait();info['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(R/'receipts/additional143_exit.json').write_text(json.dumps(info,indent=2))
    if info['returncode']:raise RuntimeError(info)

if __name__=='__main__':main()
