"""Wait for actual2000 receipt, then run native capacity/train/eval on GPU0.

Does not modify the native optimizer, active Pick PPO, or reference producers.
Only its own newly launched process groups may be stopped by this controller.
"""
import argparse
import csv
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'stage3_supermimic29_20261103';S=R/'stage3_pick29_20261103'
N=D/'native_universal_20261104';B=R/'stage3_supported_references_20261104'
Q=R/'stage3_skill_sequence_20261103';GiB=1024**3;MiB=1024**2
FLOOR=8*GiB+356*MiB;DATA_FLOOR=8*GiB+456*MiB
sys.path[:0]=[str(D/'src'),str(S/'src')]
from native_tracker2000_worker_v1 import read,sha,atomic,dataset,budget_plan,LAST,EARLY
from stage3_pick_parallel_controller import resources,stop_group
from stage3_pick500_replacement_queue import snapshot

def budget():
    full=N/'resume_8912_v1/logs/last.pt'
    small=N/'benchmark_8912_v1/logs/ckpt_000010.pt'
    assert full.is_file() and small.is_file()
    f=math.ceil(full.stat().st_size*1.15/MiB)*MiB
    s=math.ceil(small.stat().st_size*1.15/MiB)*MiB
    main_numbers=list(range(2000,LAST,2000))+[LAST]
    return {'full_reserve_each':f,'small_reserve_each':s,
            'main_small_checkpoints':len(main_numbers),
            'native_total_output_reserve':2*f+(len(main_numbers)+1)*s+f+s+100*MiB,
            'early_iteration':EARLY,'main_checkpoint_iterations':main_numbers}

def remaining_ppo_reserve():
    run=S/'runs/pick500_8912_3000_v1';status=read(run/'status.json')
    sizes=[p.stat().st_size for p in (run/'ppo').glob('model_*.pt')]
    assert sizes
    # Reserve a whole next5000-update task as well, including before its launch.
    # This prevents two independent launch checks from spending the same space.
    other=Q/'active_supported_ppo_run.json'
    if other.exists():
        current=Path(read(other)['run_dir'])
        sizes += [p.stat().st_size for p in (current/'ppo').glob('model_*.pt')]
    unit=math.ceil(max(sizes)*1.5/MiB)*MiB
    pick_n=0 if status['status']=='ppo_completed' else math.ceil(max(0,3000-status['policy_updates'])/50)+2
    return max(pick_n,math.ceil(5000/50)+3)*unit+100*MiB

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    assert a.output.resolve().is_relative_to(N.resolve())
    a.output.mkdir(exist_ok=False);began=time.monotonic();child=None
    lock=(N/'native2000_controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    worker=D/'src/native_tracker2000_worker_v1.py';evaluator=D/'src/native_tracker2000_eval_v1.py'
    spec=read(N/'environment_spec.json')['spec']
    sources=[Path(__file__),worker,evaluator,N/'environment_spec.json']
    sources+=list((N/'source/supermimic').glob('*.py'))
    bound={str(x):sha(x) for x in sources}
    (a.output/'source_binding.json').open('x').write(json.dumps(bound,indent=2))
    reserve=budget();plan=budget_plan()
    state={'status':'waiting_final2000_receipt','pid':os.getpid(),
           'formal_training_started':False,'budget':plan,'checkpoint_reserve':reserve,
           'output':str(a.output),'cache_cap_bytes':2*GiB,
           'cache_budget_change':'2GiB cap vs previous1GiB; launch checks actual shared-memory free space',
           'source_binding':str(a.output/'source_binding.json')}
    def save(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began);atomic(a.output/'status.json',state)
    def check_sources():
        assert all(sha(k)==h for k,h in bound.items()),'bound source changed; preserve prefix'
    cache=Path('/dev/shm/dcv2_native_smoke_author_v1')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':spec['gpu_uuid'],
         'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
         'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
         'MAX_JOBS':'1','TORCHINDUCTOR_COMPILE_THREADS':'1','NVRTC_DISABLE_PCH':'1',
         'CUDA_CACHE_DISABLE':'1','XDG_CACHE_HOME':str(cache/'xdg'),
         'WARP_CACHE_PATH':str(cache/'warp'),'TRITON_CACHE_DIR':str(cache/'triton'),
         'TORCHINDUCTOR_CACHE_DIR':str(cache/'inductor'),'MPLCONFIGDIR':str(cache/'mpl'),
         'TMPDIR':str(cache),'TMP':str(cache),'TEMP':str(cache)}
    def shared_ready(required_disk):
        total=snapshot();gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free',
                     '--format=csv,noheader,nounits'],text=True)
        available={x.split(',')[0].strip():int(x.split(',')[1].strip())*MiB for x in gpu.splitlines()}
        return (total['free_bytes']>=required_disk and available[spec['gpu_uuid']]>=12*GiB
                and total['owned_gpu_bytes']+12*GiB<=24*GiB
                and total['owned_rss_bytes']+16*GiB<=64*GiB),total,available[spec['gpu_uuid']]
    def wait_resources(required_disk,label):
        while True:
            check_sources();ok,total,gpu_free=shared_ready(required_disk)
            save(status=label,resources=total,required_start_free_bytes=required_disk,gpu_free_bytes=gpu_free)
            assert cache.is_dir() and cache.stat().st_uid==os.getuid()
            assert shutil.disk_usage('/dev/shm').free>=4*GiB,'shared-memory cache reserve'
            if ok:break
            assert time.monotonic()-began<48*3600,'bounded resource wait; inspect preserved status'
            time.sleep(15)
    def run(label,command,output,expected,deadline):
        nonlocal child
        check_sources();log=(a.output/(label+'.log')).open('x')
        child=subprocess.Popen(command,env=env,cwd=str(N/'source'),stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (a.output/(label+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,
            'command':command,'source_binding':str(a.output/'source_binding.json')},indent=2))
        phase=time.monotonic();done_since=None;save(status=label,child_pid=child.pid)
        while child.poll() is None:
            own=resources(child.pid);total=snapshot()
            save(resources=own,aggregate_resources=total)
            assert own['gpu_bytes']<=12*GiB and own['rss_bytes']<=32*GiB
            assert total['owned_gpu_bytes']<=24*GiB and total['owned_rss_bytes']<=64*GiB
            assert total['free_bytes']>=FLOOR
            cache_bytes=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
            assert cache_bytes<=2*GiB,'private cache cap'
            assert time.monotonic()-phase<deadline
            if (output/'status.json').exists():
                leaf=read(output/'status.json')
                if leaf['status']=='failed_preserved':raise RuntimeError(leaf.get('error'))
                if leaf['status']==expected:
                    if done_since is None:done_since=time.monotonic()
                    if time.monotonic()-done_since>40:
                        stop_group(child.pid);save(cleanup_only_group_released=True);break
            csvpath=output/'logs/train_log.csv'
            if csvpath.exists():
                with csvpath.open() as f:rows=list(csv.DictReader(f))
                complete=[r for r in rows if r.get('iteration') and r.get('lr')]
                if complete:state['native_metrics_tail']=complete[-1]
            time.sleep(5)
        child.wait();log.close();child=None
        result=read(output/'status.json');assert result['status']==expected,'native stage did not complete'
        return result
    try:
        save()
        if a.prepare_only:
            assert 'torch' not in sys.modules
            save(status='native2000_controller_prepared_not_launched',runtime_validated=False,GPU_work=False)
            return
        while not (B/'mixed2000_final_v1/completion_receipt.json').exists():
            check_sources();bulk=read(B/'bulk_v1/controller_status.json')
            mixed=read(B/'mixed2000_completion_v1/status.json')
            assert bulk['status']!='failed_preserved' and mixed['status']!='failed_preserved'
            save(status='waiting_final2000_receipt',reference_counts=bulk['native_admitted_counts'])
            assert time.monotonic()-began<48*3600
            time.sleep(15)
        receipt,manifest,msha=dataset();save(manifest_sha256=msha,reference_count=2000)
        wait_resources(DATA_FLOOR+150*MiB,'waiting_native2000_capacity_resources')
        capacity=a.output/'capacity_8912_v1'
        cap=run('capacity',[spec['python'],'-I','-B','-u',str(worker),'--phase','capacity',
                '--output',str(capacity)],capacity,'native2000_capacity_passed',2400)
        cap_path=capacity/'status.json'
        required=max(11*GiB,FLOOR+reserve['native_total_output_reserve']+remaining_ppo_reserve())
        wait_resources(required,'waiting_native2000_formal_disk_and_checkpoint_budget')
        gatepath=N/'all_four_reference_gate.json';gate=read(gatepath)
        (a.output/'gate_before.json').open('x').write(json.dumps(gate,indent=2))
        gate.update(status='four_task2000_capacity_and_formal_resource_gates_passed',training_allowed=True,
                    reference_count=2000,per_task=receipt['per_task'],manifest_sha256=msha,
                    capacity_report=str(cap_path),capacity_report_sha256=sha(cap_path),
                    formal_guard=str(a.output/'status.json'),source_binding=str(a.output/'source_binding.json'))
        atomic(gatepath,gate)
        pointer={'status':'native2000_early_formal_process_starting','controller_pid':os.getpid(),
                 'controller_status':str(a.output/'status.json'),'formal_training_started':False,
                 'manifest_sha256':msha,'reference_count':2000,'gpu':0,'budget':plan}
        atomic(D/'active_native_universal_run.json',pointer)
        early=a.output/'early600_v1';mainout=a.output/'main_resume_v1'
        result=run('early600',[spec['python'],'-I','-B','-u',str(worker),'--phase','early',
                   '--output',str(early)],early,'native2000_early_checkpoint_verified',7200)
        assert result['optimizer_updates']['opt_pi']>0
        save(formal_training_started=True,early_checkpoint=result['checkpoint'])
        pointer.update(status='native2000_formal_running',formal_training_started=True,
                       early_checkpoint=result['checkpoint'],run_dir=str(mainout))
        atomic(D/'active_native_universal_run.json',pointer)
        result=run('main',[spec['python'],'-I','-B','-u',str(worker),'--phase','main',
                   '--early-output',str(early),'--output',str(mainout)],mainout,
                   'native2000_training_budget_completed_evaluation_pending',24*3600)
        assert result['real_vector_steps']==72000 and result['manifest_sha256']==msha
        evaluation=a.output/'evaluation_all2000_v1'
        wait_resources(FLOOR+100*MiB,'waiting_native2000_evaluation_resources')
        run('evaluation',[spec['python'],'-I','-B','-u',str(evaluator),'--training-output',str(mainout),
            '--output',str(evaluation)],evaluation,'native2000_body_tracking_evaluation_completed',3600)
        report=evaluation/'report.json';complete={'status':'native2000_training_and_body_tracking_evaluation_completed',
            'manifest_sha256':msha,'real_environment_transitions':641664000,'training':result,
            'evaluation_report':str(report),'evaluation_report_sha256':sha(report),
            'per_task':read(report)['per_task'],'object_task_success_claimed':False,
            'source_binding':str(a.output/'source_binding.json')}
        target=Q/'comparisons/native_universal_four_tasks2000_v1.json';assert not target.exists()
        atomic(target,complete);save(status=complete['status'],completion=str(target))
        pointer.update(status=complete['status'],completion=str(target));atomic(D/'active_native_universal_run.json',pointer)
    except BaseException as error:
        save(status='failed_preserved',error=repr(error),traceback=traceback.format_exc())
        if child and child.poll() is None:stop_group(child.pid)
        raise
    finally:lock.close()

if __name__=='__main__':main()
