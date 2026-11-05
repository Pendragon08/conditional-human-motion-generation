"""Finish the Pick500 library, verify the real scene, then authorized PPO/eval.

Every phase consumes measured/hash-bound artifacts. This controller never starts
an unsupported task, lowers resource floors, deletes files or restarts old jobs.
"""
import fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104';Q=R/'stage3_skill_sequence_20261103'
W=O/'completion_controller_v1';P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
sys.path.insert(0,str(S/'src'))
from stage3_pick500_replacement_queue import snapshot
FLOOR=8*1024**3+456*1024**2;RUN_FLOOR=8*1024**3+356*1024**2
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,j):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(p)

def main():
    W.mkdir(exist_ok=False);state={'status':'waiting_reference500_and_source_audits','pid':os.getpid(),'formal_training_started':False}
    lock=(O/'completion_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic();child=None;log=None
    def save(**kwargs):
        state.update(kwargs,elapsed_seconds=time.monotonic()-began);atomic(W/'status.json',state)
    codes={n:sha(S/'src'/n) for n in ['stage3_pick500_finalize_library.py','stage3_pick500_ppo.py','stage3_pick500_evaluate.py',
        'stage3_pick500_env.py','stage3_pick29_env.py','stage3_pick29_core.py']}
    (W/'source_binding.json').open('x').write(json.dumps(codes,indent=2))
    def guard(floor=FLOOR):
        s=snapshot();save(resources=s)
        assert s['free_bytes']>=floor,'applicable disk floor reached'
        assert s['owned_gpu_bytes']<=12*1024**3 and s['owned_rss_bytes']<=32*1024**3,'owned job GPU/RSS protection'
    def progress(phase,run=None,evaluation=None):
        with (Q/'controller.lock').open('a') as qlock:
            fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            j=read(Q/'progress.json');assert j['tasks'][0]['id']=='pick'
            j.update(current_index=0,current_task='pick',status=phase,controller_pid=os.getpid(),
                active_pick500_controller=str(W/'status.json'))
            if run:j['tasks'][0].update(status=phase,run_dir=str(run),target_references=500,total_updates=3000)
            if evaluation:j['tasks'][0].update(status='budget_completed_evaluation_archived',observed_success_ratio=evaluation['success_ratio'])
            atomic(Q/'progress.json',j)
    spec=read(R/'stage3_supermimic29_20261103/runtime_bridge_spec.json')
    def run_phase(name,command,env,cwd,expected,validate,floor=FLOOR,timeout=600):
        nonlocal child,log
        for n,h in codes.items():assert sha(S/'src'/n)==h,'bound source changed; inspect instead of launching'
        guard(floor);save(status=name,command=command)
        assert not (W/(name+'_launch.json')).exists()
        log=(W/(name+'.log')).open('x')
        child=subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (W/(name+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,'command':command,'expected_report':str(expected)}))
        if name=='formal_pick500_ppo':save(formal_process_pid=child.pid)
        start=time.monotonic();finished_at=None
        while child.poll() is None:
            guard(floor)
            if expected.exists():
                try:r=read(expected)
                except json.JSONDecodeError:r={}
                if r.get('status') in ['failed_preserved','failed']:raise RuntimeError(name+' failed; preserve original report/log')
                if name=='formal_pick500_ppo' and r.get('policy_updates',0)>0:
                    first_actual_update=not state.get('formal_training_started',False)
                    save(formal_training_started=True,policy_updates=r['policy_updates'])
                    if first_actual_update:progress('pick500_formal_PPO_running',run=expected.parent)
                if name=='formal_pick500_ppo' and r.get('status')=='ppo_completed':
                    assert r['policy_updates']==3000 and sha(r['final_checkpoint'])==r['final_checkpoint_sha256']
                    if finished_at is None:finished_at=time.monotonic()
                    if time.monotonic()-finished_at>40:
                        os.killpg(child.pid,signal.SIGTERM);state['completed_ppo_cleanup_released']=True;break
            assert time.monotonic()-start<timeout,'bounded phase timeout'
            time.sleep(5)
        try:code=child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        log.close();log=None
        assert expected.exists(),name+' exited without expected complete report'
        result=read(expected);validate(result)
        assert code==0 or (name=='formal_pick500_ppo' and state.get('completed_ppo_cleanup_released')),name+' child exited unsuccessfully'
        (W/(name+'_exit.json')).open('x').write(json.dumps({'returncode':code,'report':str(expected),'sha256':sha(expected),'validated':True}))
        child=None;return result
    try:
        save()
        while True:
            guard()
            native=read(O/'native_drain_controller_v1/status.json');replacement=read(O/'replacement_queue128_v1/status.json')
            assert not native['status'].startswith('failed') and not replacement['status'].startswith('failed'),'source/admission failure requires inspection'
            library=read(O/'library_progress.json');save(native_admitted=library['native_admitted'])
            if library['native_admitted']>=500 and (O/'replacement_queue128_v1/report.json').exists():break
            if native['status']=='initial_sources_drained_replacement_sampling_required':
                save(status='waiting_additional_new_seed_replacements',deficit=max(0,500-library['native_admitted']))
            time.sleep(15)
        replacement=read(O/'replacement_queue128_v1/report.json')
        assert replacement['unique_IDs_noise_features'] and replacement['candidate_count']==628
        assert read(O/'replacement_queue128_v1/original500_cross_audit.json')['passed']
        audits=[O/'pilot_candidate_audit.json',O/'replacement_queue128_v1/remaining496_candidate_audit.json',
                O/'replacement_queue128_v1/replacement128_candidate_audit.json']
        for path in audits:assert read(path)['passed']
        final=O/'final_library_v1';assert not final.exists(),'preserved final output exists; inspect rather than rerun'
        command=[P,'-u',str(S/'src/stage3_pick500_finalize_library.py'),'--output-dir',str(final)]
        for path in audits:command.extend(['--candidate-audit',str(path)])
        cpu=run_phase('final_library_cpu_audit',command,{**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},str(S),final/'cpu_audit.json',
            lambda r:require(r['status']=='cpu_integrity_passed_native_loader_pending' and r['reference_count']==500),timeout=600)
        manifest=Path(cpu['manifest']);manifest_sha=sha(manifest);assert manifest_sha==cpu['manifest_sha256']
        output=O/'ppo_capacity8912_v1';assert not output.exists()
        command=[spec['python'],'-u',str(S/'src/stage3_pick500_ppo.py'),'--headless','--device','cuda:1',
            '--capacity-test','--reference-manifest',str(manifest),'--output-dir',str(output),'--num-envs','8912','--steps','128']
        capacity=run_phase('final_bank_8912_capacity',command,{**os.environ,**spec['env']},spec['cwd'],output/'status.json',
            lambda r:require(r['status']=='capacity_completed' and r['reference_manifest_sha256']==manifest_sha and r['num_envs']==8912
                and r['steps']>=128 and r['reference_count']==500 and r['object_reset_error_m']<1e-5 and r['actor_reference_change_error']==0),timeout=600)
        assert sha(manifest)==manifest_sha
        receipt={'status':'passed','reference_count':500,'manifest':str(manifest),'manifest_sha256':manifest_sha,
            'cpu_audit':str(final/'cpu_audit.json'),'cpu_audit_sha256':sha(final/'cpu_audit.json'),
            'actual_default_ReferenceBank500_loaded':True,'native_capacity':str(output/'status.json'),'native_capacity_sha256':sha(output/'status.json'),
            'object_reset_error_m':capacity['object_reset_error_m'],'actor_reference_change_error':capacity['actor_reference_change_error'],
            'scope':'source/kinematic/native-reference-loader admission; not PD tracking or Pick success','formal_PPO_started_at_receipt':False}
        (O/'final_library_audit.json').open('x').write(json.dumps(receipt,indent=2))
        atomic(O/'active_reference_bank500.json',{'manifest':str(manifest),'sha256':manifest_sha,'audit':str(O/'final_library_audit.json'),'audit_sha256':sha(O/'final_library_audit.json')})
        progress('pick500_library_admitted_waiting_formal_resources');save(status='pick500_library_admitted_waiting_formal_resources',reference_manifest=str(manifest),reference_manifest_sha256=manifest_sha)
        smoke=read(O/'ppo_integration_v1/status.json');assert smoke['status']=='ppo_completed' and smoke['checkpoint_roundtrip_verified']
        evaluate_smoke=read(O/'evaluation_entry4_v1/report.json');assert evaluate_smoke['status']=='integration_evaluation_completed' and evaluate_smoke['policy_state_hash_unchanged']
        checkpoint_budget=math.ceil(smoke['largest_checkpoint_bytes']*1.25/1024**2)*1024**2
        need=max(11*1024**3,RUN_FLOOR+63*checkpoint_budget)
        while shutil.disk_usage('/data').free<need:
            guard(RUN_FLOOR);save(status='pick500_ready_waiting_training_disk_budget',required_start_free_bytes=need,
                free_bytes=shutil.disk_usage('/data').free,shortfall_bytes=need-shutil.disk_usage('/data').free)
            time.sleep(60)
        run=S/'runs/pick500_8912_3000_v1';assert not run.exists()
        active={'run_dir':str(run),'reference_manifest':str(manifest),'reference_manifest_sha256':manifest_sha,'total_updates':3000,
            'num_envs':8912,'controller_pid':os.getpid(),'controller_status':str(W/'status.json'),'baseline_preserved':True,
            'source_sha256':codes,'required_start_free_bytes':need}
        atomic(O/'active_pick500_run.json',active);progress('pick500_formal_PPO_launch_preflight',run=run)
        save(run_dir=str(run))
        command=[spec['python'],'-u',str(S/'src/stage3_pick500_ppo.py'),'--headless','--device','cuda:1','--reference-manifest',str(manifest),
            '--output-dir',str(run),'--num-envs','8912','--max-iterations','3000']
        training=run_phase('formal_pick500_ppo',command,{**os.environ,**spec['env']},spec['cwd'],run/'status.json',
            lambda r:require(r['status']=='ppo_completed' and r['policy_updates']==3000 and r['reference_count']==500
                and r['reference_manifest_sha256']==manifest_sha and sha(r['final_checkpoint'])==r['final_checkpoint_sha256']),
            floor=RUN_FLOOR,timeout=48*3600)
        checkpoint=Path(training['final_checkpoint']);evaluation=Q/'evaluations/pick500_final_3000_v1';assert not evaluation.exists()
        command=[spec['python'],'-u',str(S/'src/stage3_pick500_evaluate.py'),'--headless','--device','cuda:1','--checkpoint',str(checkpoint),
            '--training-run',str(run),'--reference-manifest',str(manifest),'--output-dir',str(evaluation),'--num-envs','100','--steps','500']
        result=run_phase('pick500_final100_evaluation',command,{**os.environ,**spec['env']},spec['cwd'],evaluation/'report.json',
            lambda r:require(r['status']=='evaluation_completed' and r['final_evaluation'] and r['num_envs']==100 and r['reference_count']==500
                and r['policy_updates_during_evaluation']==0 and r['policy_state_hash_unchanged']
                and r['checkpoint_sha256']==training['final_checkpoint_sha256'] and r['reference_bank_sha256']==manifest_sha),floor=RUN_FLOOR,timeout=1200)
        completion={'status':'training_budget_completed_evaluation_archived','task':'pick','references':500,'updates':3000,
            'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'reference_manifest':str(manifest),'reference_manifest_sha256':manifest_sha,
            'evaluation':str(evaluation/'report.json'),'evaluation_sha256':sha(evaluation/'report.json'),
            'observed_success_ratio':result['success_ratio'],'observed_success_count':result['success_count'],
            'success_definition':result['success_definition'],'training_budget_is_not_success':True,'scope':'simulation only',
            'next_step':'same-library SuperMimic comparison retained separately; remaining active skills need task-specific adapters, not old8-task queue'}
        (Q/'completions/pick500_reference500_v1.json').open('x').write(json.dumps(completion,indent=2))
        progress('pick500_budget_completed_evaluation_archived',run=run,evaluation=result)
        (W/'report.json').open('x').write(json.dumps(completion,indent=2));save(**completion)
    except BaseException as e:
        save(status='failed_preserved',error=repr(e))
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        if log is not None:log.close()
        lock.close()

def require(value):
    if not value:raise RuntimeError('measured phase receipt did not satisfy required conditions')

if __name__=='__main__':main()
