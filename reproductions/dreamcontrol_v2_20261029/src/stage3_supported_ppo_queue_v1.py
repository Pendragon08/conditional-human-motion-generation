"""Independent skill PPO continuation on GPU1, concurrent with native GPU0.

Validate each new task with a real4-env PPO first. After existing Pick and its
evaluation, each task waits its own500 scene-admitted references, then executes
CPU bank verification,8912 capacity,5000 PPO updates and100-env evaluation.
"""
import argparse,fcntl,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
B=R/'stage3_supported_references_20261104';S=R/'stage3_pick29_20261103'
D=R/'stage3_supermimic29_20261103';N=D/'native_universal_20261104';Q=R/'stage3_skill_sequence_20261103'
J=B/'task_ppo_20261104';P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
sys.path[:0]=[str(S/'src'),str(D/'src')]
from stage3_pick_parallel_controller import resources,stop_group,read,sha,atomic
from stage3_pick500_replacement_queue import snapshot
GiB=1024**3;MiB=1024**2;FLOOR=8*GiB+356*MiB
TASKS=['pour','bimanual_box_lift','box_push_pull']

def native_remaining_reserve():
    pointer=N/'active_native2000_controller.json'
    if not pointer.exists():return 2300*MiB
    doc=read(pointer);status=read(doc['controller_status'])
    if status['status']=='native2000_training_and_body_tracking_evaluation_completed':return 100*MiB
    # Budget the full pending native job even before its formal launch.
    used=sum(p.stat().st_size for p in Path(status['output']).rglob('*') if p.is_file())
    return max(300*MiB,status['checkpoint_reserve']['native_total_output_reserve']-used)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot-only',action='store_true');a=parser.parse_args()
    assert a.output.resolve().is_relative_to(J.resolve());a.output.mkdir(exist_ok=False)
    lock=(J/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=read(S/'env_spec_v2.json');env={**os.environ,**spec['env'],
         'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1'}
    env['PYTHONPATH']=str(B/'src')+':'+env['PYTHONPATH'];env.pop('CUDA_VISIBLE_DEVICES',None)
    worker=B/'src/stage3_supported_ppo_train_v1.py';builder=B/'src/stage3_supported_ppo_manifest_v1.py'
    sources=[Path(__file__),worker,builder,B/'src/stage3_supported_ppo_env_v1.py',
             B/'src/stage3_supported_ppo_core_v1.py',S/'src/stage3_pick29_env.py',S/'src/stage3_pick29_core.py']
    bound={str(p):sha(p) for p in sources}
    (a.output/'source_binding.json').open('x').write(json.dumps(bound,indent=2))
    state={'status':'validating_new_task_pilots','pid':os.getpid(),'formal_training_started':False,
           'PPO_gpu':1,'native_gpu':0,'task_order':TASKS,'task_results':{},'current_pick_untouched':True}
    child=None;began=time.monotonic()
    def save(**kw):state.update(kw,elapsed_seconds=time.monotonic()-began);atomic(a.output/'status.json',state)
    def check():assert all(sha(k)==v for k,v in bound.items()),'queue source changed; preserve prefix'
    def wait_space(required,label):
        while True:
            check();s=snapshot();save(status=label,required_start_free_bytes=required,resources=s)
            if s['free_bytes']>=required:break
            assert time.monotonic()-began<96*3600;time.sleep(15)
    def wait_gpu():
        while True:
            s=snapshot();gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True)
            if int(gpu.strip())*MiB>=12*GiB and s['owned_gpu_bytes']+12*GiB<=24*GiB and s['owned_rss_bytes']+16*GiB<=64*GiB:break
            save(status='waiting_PPO_gpu1_resource_room',resources=s);time.sleep(15)
    def run(label,command,output,terminal,timeout,cwd):
        nonlocal child
        check();log=(a.output/(label+'.log')).open('x')
        child=subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,
                               stderr=subprocess.STDOUT,start_new_session=True)
        (a.output/(label+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,'command':command},indent=2))
        start=time.monotonic();finished=None;save(status=label,child_pid=child.pid)
        while child.poll() is None:
            own=resources(child.pid);total=snapshot();save(resources=own,aggregate_resources=total)
            assert own['gpu_bytes']<=12*GiB and own['rss_bytes']<=32*GiB
            assert total['owned_gpu_bytes']<=24*GiB and total['owned_rss_bytes']<=64*GiB
            assert total['free_bytes']>=FLOOR and time.monotonic()-start<timeout
            if (output/'status.json').exists():
                leaf=read(output/'status.json')
                if leaf['status']=='failed_preserved':raise RuntimeError(leaf.get('error'))
                if leaf.get('formal_training_started'):
                    state['formal_training_started']=True
                    pointer=Q/'active_supported_ppo_run.json'
                    if pointer.exists() and read(pointer).get('run_dir')==str(output):
                        ptr=read(pointer);ptr.update(policy_updates=leaf.get('policy_updates'),formal_training_started=True)
                        atomic(pointer,ptr)
                if leaf['status']==terminal:
                    if finished is None:finished=time.monotonic()
                    if time.monotonic()-finished>40:stop_group(child.pid);break
            time.sleep(3)
        code=child.wait();log.close();child=None
        result=read(output/'status.json') if (output/'status.json').exists() else read(output/'report.json')
        assert result['status']==terminal,'task stage failed; inspect preserved logs'
        return result
    def command(task,mode,bank,output,num_envs,iterations=5000,extra=None):
        return [spec['smoke_command'][0],'-u',str(worker),'--headless','--device','cuda:1',
                '--task',task,'--mode',mode,'--reference-manifest',str(bank),'--output-dir',str(output),
                '--num-envs',str(num_envs),'--max-iterations',str(iterations)]+(extra or [])
    try:
        save()
        # Tests are deliberately separate from formal budgets and never repeated.
        for task in TASKS:
            root=J/task;bank=root/'pilot_bank_v1/reference_bank.json';test=root/'integration_v1'
            assert bank.exists()
            wait_space(8*GiB+456*MiB,'waiting_task_integration_disk')
            wait_gpu();result=run(task+'_integration',command(task,'integration',bank,test,4,2),
                         test,'ppo_completed',1800,spec['cwd'])
            assert result['policy_updates']==2 and result['checkpoint_roundtrip_verified']
            state['task_results'][task]={'integration':str(test/'status.json'),'formal_started':False};save()
        if a.pilot_only:
            save(status='three_task_PPO_integrations_passed_formal_not_started');return
        pick=S/'runs/pick500_8912_3000_v1/status.json'
        peval=Q/'evaluations/pick500_final_3000_v1/report.json'
        while not peval.exists():
            check();leaf=read(pick);assert leaf['status']!='failed_preserved'
            save(status='waiting_existing_Pick_PPO_and100env_evaluation',pick_updates=leaf['policy_updates'])
            time.sleep(15)
        assert read(pick)['status']=='ppo_completed' and read(pick)['policy_updates']==3000
        assert read(peval)['status']=='evaluation_completed' and read(peval)['num_envs']==100
        for task in TASKS:
            root=J/task;state['active_task']=task
            while True:
                check();bulk=read(B/'bulk_v1/controller_status.json')
                assert bulk['status']!='failed_preserved'
                count=bulk['native_admitted_counts'][task]
                save(status='waiting_task500_scene_admissions',task_reference_count=count)
                if count>=500:break
                time.sleep(15)
            bankdir=root/'final500_v1';bank=bankdir/'reference_bank.json'
            wait_space(8*GiB+456*MiB,'waiting_task_bank_disk')
            run(task+'_bank',[P,'-B','-u',str(builder),'--task',task,'--count','500',
                '--output-dir',str(bankdir)],bankdir,'task_reference_bank_cpu_verified',1800,str(B))
            wait_gpu();capacity=root/'capacity8912_v1'
            cap=run(task+'_capacity',command(task,'capacity',bank,capacity,8912),capacity,'capacity_completed',1800,spec['cwd'])
            test=read(root/'integration_v1/status.json');unit=math.ceil(test['largest_checkpoint_bytes']*1.5/MiB)*MiB
            iterations=next(t.get('iteration_target',read(Q/'sequence.json')['formal_iteration_target'])
                            for t in read(Q/'sequence.json')['tasks'] if t['id']==task)
            assert iterations==5000,'explicit per-task budget change requires a fresh bound run configuration'
            own_reserve=(math.ceil(iterations/50)+3)*unit+100*MiB
            required=max(11*GiB,FLOOR+own_reserve+native_remaining_reserve())
            wait_space(required,'waiting_task_PPO_and_native_combined_checkpoint_budget');wait_gpu()
            ready={'status':'task_ppo_formal_preflight_passed','task':task,'manifest_sha256':sha(bank),
                   'capacity_status':str(capacity/'status.json'),'capacity_status_sha256':sha(capacity/'status.json'),
                   'integration_status_sha256':sha(root/'integration_v1/status.json'),
                   'iteration_target':iterations,'required_start_free_bytes':required,
                   'source_binding':str(a.output/'source_binding.json'),'PPO_checkpoint_reserve':own_reserve,
                   'native_remaining_reserve':native_remaining_reserve()}
            ready_path=root/'formal_preflight_v1.json';assert not ready_path.exists();atomic(ready_path,ready)
            output=root/'ppo_8912_5000_v1';pointer={'task':task,'run_dir':str(output),'gpu':1,
                'controller_pid':os.getpid(),'formal_training_started':False,'max_iterations':iterations,
                'manifest_sha256':sha(bank),'controller_status':str(a.output/'status.json')}
            atomic(Q/'active_supported_ppo_run.json',pointer)
            train=run(task+'_formal',command(task,'formal',bank,output,8912,iterations,
                ['--ready-report',str(ready_path)]),output,'ppo_completed',48*3600,spec['cwd'])
            assert train['policy_updates']==iterations and train['checkpoint_roundtrip_verified']
            evaluation=root/'evaluation100_v1'
            wait_space(FLOOR+100*MiB,'waiting_task_evaluation_disk');wait_gpu()
            report=run(task+'_evaluate',command(task,'evaluate',bank,evaluation,100,iterations,
                 ['--training-run',str(output)]),evaluation,'task_ppo_evaluation_completed',1800,spec['cwd'])
            complete={'status':'independent_task_PPO_and100env_evaluation_completed','task':task,
                'training_status':str(output/'status.json'),'training_status_sha256':sha(output/'status.json'),
                'iteration_target':iterations,'evaluation_report':str(evaluation/'report.json'),
                'evaluation_report_sha256':sha(evaluation/'report.json'),'success_ratio':report['success_ratio'],
                'success_count':report['success_count'],'scope':report['contract']['scope']}
            path=Q/('completions/'+task+'_ppo_reference500_5000_v1.json');assert not path.exists();atomic(path,complete)
            state['task_results'][task].update(completion=str(path),success_ratio=report['success_ratio']);save()
        save(status='all_three_independent_PPO_tasks_and_evaluations_completed')
    except BaseException as error:
        save(status='failed_preserved',error=repr(error),traceback=traceback.format_exc())
        if child and child.poll() is None:stop_group(child.pid)
        raise
    finally:lock.close()

if __name__=='__main__':main()
