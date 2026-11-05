"""Adopt the running Pick PPO without restarting it; prepare a parallel GPU0 comparison.

The newer user request authorizes independent concurrent jobs. Per-job limits
remain12GiB GPU/32GiB RSS; the two-job aggregate is24GiB/64GiB. Disk floors and
all immutable task/training code stay unchanged. Other users' jobs are read only.
"""
import fcntl,hashlib,json,math,os,shutil,signal,subprocess,time,traceback
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';O=S/'reference_scale500_20261104';D=R/'stage3_supermimic29_20261103';Q=R/'stage3_skill_sequence_20261103'
W=D/'parallel_pick500_20261104';C=W/'controller_v1';RUN=S/'runs/pick500_8912_3000_v1'
MAN=O/'final_library_v1/reference_bank.json';SM=D/'runs/pick500_8912_72000_parallel_v1'
FLOOR=8*1024**3+356*1024**2
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,j):
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(p)
def alive(pid):
    try:return Path('/proc',str(pid),'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
def processes():
    result={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,rss=,args='],universal_newlines=True).splitlines():
        row=line.strip().split(None,3)
        if len(row)==4:result[int(row[0])]={'ppid':int(row[1]),'rss':int(row[2])*1024,'args':row[3]}
    return result
def resources(root):
    table=processes();ids={root} if alive(root) else set()
    while True:
        n=len(ids);ids.update(k for k,v in table.items() if v['ppid'] in ids)
        if n==len(ids):break
    gpu={}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],universal_newlines=True,timeout=15).splitlines():
        pid,uuid,mem=[x.strip() for x in line.split(',')]
        if int(pid) in ids:
            assert mem.isdigit(),'GPU accounting unavailable'
            gpu[uuid]=gpu.get(uuid,0)+int(mem)*1024**2
    return {'pids':sorted(ids),'gpu_bytes':sum(gpu.values()),'gpu_by_uuid':gpu,'rss_bytes':sum(table[x]['rss'] for x in ids if x in table)}
def stop_group(pid):
    if alive(pid):
        assert str(R) in Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode()
        os.killpg(pid,signal.SIGTERM)
        for _ in range(40):
            if not alive(pid):break
            time.sleep(.5)
        if alive(pid):os.killpg(pid,signal.SIGKILL)
def main():
    C.mkdir(exist_ok=False)
    lock=(W/'parallel.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'status':'adopting_existing_PPO','supermimic_formal_started':False,
        'policy':'parallel independent Pick comparisons, no PPO restart','limits':{'gpu_per_job':12*1024**3,'rss_per_job':32*1024**3,'gpu_aggregate':24*1024**3,'rss_aggregate':64*1024**3},'disk_runtime_floor':FLOOR}
    def save(**kw):state.update(kw);atomic(C/'status.json',state)
    spec=read(D/'runtime_bridge_spec.json');env={**os.environ,**spec['env']}
    binding={str(p):sha(p) for p in [S/'src/stage3_pick500_ppo.py',S/'src/stage3_pick500_env.py',S/'src/stage3_pick29_env.py',S/'src/stage3_pick29_core.py',S/'src/stage3_pick500_evaluate.py',D/'src/stage3_supermimic_pick500_train.py',D/'src/stage3_supermimic_pick_core.py',D/'src/stage3_supermimic_replay.py',D/'src/stage3_supermimic_budget.py',MAN]}
    (C/'source_binding.json').open('x').write(json.dumps(binding,indent=2))
    ppo=read(RUN/'status.json');assert ppo['status']=='ppo_running' and ppo['reference_count']==500
    assert sha(MAN)==ppo['reference_manifest_sha256'] and read(O/'final_library_audit.json')['status']=='passed'
    ppo_group=os.getpgid(ppo['pid']);assert ppo_group==1436626
    old=read(O/'active_completion_controller.json');oldpid=old['pid'] if 'pid' in old else old['controller_pid']
    assert oldpid==1427653
    assert 'stage3_pick500_completion_controller.py' in Path('/proc',str(oldpid),'cmdline').read_bytes().decode()
    assert sha(S/'src/stage3_pick500_completion_controller.py')=='c7a3b1c758528a323a952090a7471461e8fe63b1f7570a46b22582be9478d9ef'
    save(status='guard_active_handoff_pending',ppo_process_group=ppo_group,ppo_python_pid=ppo['pid'],old_controller_pid=oldpid)
    # TERM of this Python-only supervisor has no signal handler and does not
    # raise its BaseException branch; separately sessioned PPO keeps running.
    os.kill(oldpid,signal.SIGTERM)
    for _ in range(20):
        if not alive(oldpid):break
        time.sleep(.2)
    assert not alive(oldpid) and alive(ppo['pid'])
    (C/'handoff_receipt.json').open('x').write(json.dumps({'previous_controller':oldpid,'new_controller':os.getpid(),
        'ppo_pid_unchanged':ppo['pid'],'ppo_process_group_unchanged':ppo_group,'ppo_update_at_handoff':ppo['policy_updates'],
        'previous_controller_source_sha256':sha(S/'src/stage3_pick500_completion_controller.py'),'running_training_source_unchanged':binding},indent=2))
    active=read(O/'active_completion_controller.json');active.update(pid=os.getpid(),controller_pid=os.getpid(),
        controller_dir=str(C),source=str(Path(__file__)),source_sha256=sha(__file__),status='parallel_guard_running',
        adopts_existing_PPO=True,previous_controller=oldpid,resource_policy=state['limits'])
    atomic(O/'active_completion_controller.json',active)
    active=read(O/'active_pick500_run.json');active.update(controller_pid=os.getpid(),controller_status=str(C/'status.json'),parallel_comparison=True)
    atomic(O/'active_pick500_run.json',active)
    atomic(W/'active_controller.json',{'pid':os.getpid(),'status':str(C/'status.json'),'source':str(Path(__file__)),'source_sha256':sha(__file__)})
    save(status='PPO_running_SuperMimic_preparation')
    child=None;handle=None;phase=None;phase_started=None;phase_done=None;next_phase='integration';sm_failed=False;ppo_complete=False
    ppo_evaluated=False;ppo_finished=None;sm_finished=None
    def start(name,command,report,terminal):
        nonlocal child,handle,phase,phase_started,phase_done
        for p,h in binding.items():assert sha(p)==h,'bound source changed'
        assert not (C/(name+'_launch.json')).exists()
        phase={'name':name,'report':str(report),'terminal':terminal};phase_started=time.monotonic();phase_done=None
        handle=(C/(name+'.log')).open('x')
        child=subprocess.Popen(command,env=env,cwd=spec['cwd'],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        (C/(name+'_launch.json')).open('x').write(json.dumps({'pid':child.pid,'command':command,'report':str(report),'phase':name},indent=2))
        save(supermimic_phase=name,child_pid=child.pid,child_report=str(report))
    train=[spec['python'],'-u',str(D/'src/stage3_supermimic_pick500_train.py'),'--headless','--device','cuda:0','--reference-manifest',str(MAN)]
    integration=W/'integration500_v1';capacity=W/'capacity8912_v1'
    while True:
        try:
            a=resources(ppo_group);b=resources(child.pid) if child else {'gpu_bytes':0,'rss_bytes':0,'pids':[]}
            free=shutil.disk_usage('/data').free
            save(ppo_resources=a,parallel_child_resources=b,free_bytes=free,ppo_updates=read(RUN/'status.json')['policy_updates'])
            # Newly added comparison yields first; never stop another user's job.
            violation=(b['gpu_bytes']>12*1024**3 or b['rss_bytes']>32*1024**3 or
                a['gpu_bytes']+b['gpu_bytes']>24*1024**3 or a['rss_bytes']+b['rss_bytes']>64*1024**3 or free<FLOOR)
            if violation and child and phase['name']!='ppo_final100_evaluation':
                stop_group(child.pid);child=None;sm_failed=True;save(supermimic_status='resource_protected_preserved',resource_violation=True)
            if a['gpu_bytes']>12*1024**3 or a['rss_bytes']>32*1024**3:
                stop_group(ppo_group);save(status='PPO_resource_protection_stopped_preserved')
                ppo_complete=True;ppo_evaluated=True
            p=read(RUN/'status.json')
            if p['status']=='ppo_completed' and not ppo_complete:
                assert p['policy_updates']==3000 and sha(p['final_checkpoint'])==p['final_checkpoint_sha256']
                if ppo_finished is None:ppo_finished=time.monotonic()
                if not alive(ppo_group) or time.monotonic()-ppo_finished>40:
                    stop_group(ppo_group);ppo_complete=True;save(ppo_status='3000_budget_completed_evaluation_pending')
            elif p['status'].startswith('failed') or (not alive(ppo_group) and not ppo_complete):
                save(ppo_status='failed_preserved',ppo_failure=p);ppo_complete=True;ppo_evaluated=True
            if child:
                result=read(phase['report']) if Path(phase['report']).exists() else {}
                if result.get('status') in ['failed','failed_preserved']:
                    stop_group(child.pid);raise RuntimeError(phase['name']+' failed: '+str(result.get('error')))
                if result.get('status')==phase['terminal']:
                    if phase_done is None:phase_done=time.monotonic()
                    if child.poll() is not None or time.monotonic()-phase_done>40:
                        stop_group(child.pid);handle.close()
                        assert result.get('reference_count',500)==500
                        receipt={'report':phase['report'],'report_sha256':sha(phase['report']),'result':result,'process_returncode':child.poll()}
                        (C/(phase['name']+'_exit.json')).open('x').write(json.dumps(receipt,indent=2))
                        if phase['name']=='integration':
                            assert result['latest_checkpoint']['roundtrip'] and result['counters']['policy_updates']>0 and result['progress']['actual_vector_steps']==96
                            next_phase='capacity'
                        elif phase['name']=='capacity':
                            assert result['config']['num_envs']==8912
                            assert result['latest_checkpoint']['roundtrip'] and result['counters']['policy_updates']>0
                            assert result['object_reset_error_m']<1e-5 and result['actor_reference_change_error']==0
                            next_phase='formal'
                        elif phase['name']=='formal':
                            assert result['progress']['actual_vector_steps']==72000 and result['latest_checkpoint']['roundtrip']
                            save(supermimic_status='interaction_budget_completed_evaluation_pending',supermimic_final=result['latest_checkpoint'])
                            next_phase='sm_evaluation_pending_implementation';sm_finished=True
                        elif phase['name']=='ppo_final100_evaluation':
                            assert result['num_envs']==100 and result['policy_updates_during_evaluation']==0 and result['policy_state_hash_unchanged']
                            completion={'status':'training_budget_completed_evaluation_archived','task':'pick','references':500,'updates':3000,
                                'checkpoint':p['final_checkpoint'],'checkpoint_sha256':p['final_checkpoint_sha256'],
                                'reference_manifest':str(MAN),'reference_manifest_sha256':sha(MAN),
                                'evaluation':phase['report'],'evaluation_sha256':sha(phase['report']),
                                'observed_success_ratio':result['success_ratio'],'observed_success_count':result['success_count'],
                                'success_definition':result['success_definition'],'training_budget_is_not_success':True,'scope':'simulation only',
                                'next_step':'parallel same500-library SuperMimic comparison; other active skills pending task-specific adapters'}
                            (Q/'completions/pick500_reference500_v1.json').open('x').write(json.dumps(completion,indent=2))
                            ppo_evaluated=True;save(ppo_status='evaluation_archived',ppo_completion=completion)
                        child=None;phase=None
                elif child.poll() is not None:raise RuntimeError(phase['name']+' exited without completed report')
                elif time.monotonic()-phase_started>(72*3600 if phase['name']=='formal' else 2400):
                    stop_group(child.pid);raise RuntimeError(phase['name']+' time bound')
                if phase and phase['name']=='formal' and result.get('counters',{}).get('policy_updates',0)>0:
                    save(supermimic_formal_started=True,supermimic_status=result.get('status'),supermimic_progress=result.get('progress'),supermimic_counters=result['counters'])
            if not child:
                if ppo_complete and not ppo_evaluated:
                    out=Q/'evaluations/pick500_final_3000_v1'
                    cmd=[spec['python'],'-u',str(S/'src/stage3_pick500_evaluate.py'),'--headless','--device','cuda:1','--checkpoint',p['final_checkpoint'],
                        '--training-run',str(RUN),'--reference-manifest',str(MAN),'--output-dir',str(out),'--num-envs','100','--steps','500']
                    start('ppo_final100_evaluation',cmd,out/'report.json','evaluation_completed')
                elif not sm_failed and next_phase in ['integration','capacity','formal']:
                    if next_phase=='integration':
                        cmd=train+['--num-envs','4','--macro-rounds','4','--model-warmup-updates','4','--output-dir',str(integration)]
                        start('integration',cmd,integration/'status.json','interaction_budget_completed')
                    elif next_phase=='capacity':
                        cmd=train+['--capacity-test','--num-envs','8912','--macro-rounds','4','--model-warmup-updates','4','--output-dir',str(capacity)]
                        start('capacity',cmd,capacity/'status.json','interaction_budget_completed')
                    else:
                        measured=read(capacity/'status.json')['latest_checkpoint']['bytes']
                        small=math.ceil(measured*0.006/1024**2)*1024**2
                        existing=max(x.stat().st_size for x in (RUN/'ppo').glob('*.pt'))
                        remaining=max(0,63-len(list((RUN/'ppo').glob('*.pt'))))*math.ceil(existing*1.25/1024**2)*1024**2
                        need=max(11*1024**3,FLOOR+math.ceil(measured*1.15)*4+small*60+remaining+64*1024**2)
                        save(supermimic_status='capacity_passed_waiting_shared_disk_reserve',required_start_free_bytes=need,remaining_PPO_checkpoint_budget=remaining,measured_full_checkpoint_bytes=measured)
                        if free>=need:
                            cmd=train+['--formal','--num-envs','8912','--macro-rounds','3000','--model-warmup-updates','500','--output-dir',str(SM)]
                            start('formal',cmd,SM/'status.json','interaction_budget_completed')
                            atomic(D/'active_pick500_run.json',{'run_dir':str(SM),'controller_pid':os.getpid(),'controller_status':str(C/'status.json'),
                                'reference_manifest':str(MAN),'reference_manifest_sha256':sha(MAN),'device':'cuda:0','num_envs':8912,
                                'real_vector_steps':72000,'real_transitions':641664000,'algorithm':'SuperMimic-style world-model/BPTT adaptation, not PPO','parallel_PPO':str(RUN)})
            if ppo_evaluated and (sm_failed or sm_finished):save(status='parallel_training_control_completed_evaluation_followup_recorded');break
        except Exception as exc:
            if child and phase and phase['name']=='ppo_final100_evaluation':
                save(ppo_status='evaluation_failed_preserved',ppo_evaluation_error=repr(exc));ppo_evaluated=True
            else:
                sm_failed=True;save(supermimic_status='failed_preserved',supermimic_error=repr(exc),traceback=traceback.format_exc())
            if child:
                stop_group(child.pid);child=None
                if handle:handle.close()
            save(status='PPO_guard_continues_after_parallel_phase_failure')
        time.sleep(5)
if __name__=='__main__':main()
