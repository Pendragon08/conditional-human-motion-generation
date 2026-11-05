"""Sequential new-seed replacements after the original500 candidates finish.

Never relaunches original jobs, changes their code or bypasses reference admission.
The existing native controller consumes registered new outputs independently.
"""
import ast,fcntl,hashlib,json,os,shutil,signal,subprocess,time
from pathlib import Path

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103';D=R/'stage2_prior29_20261103';O=S/'reference_scale500_20261104'
W=O/'replacement_queue128_v1';P='/data/code/ouyanghao/oyh/reactivebfm-reproduction-20261021/env/bin/python'
FLOOR=8*1024**3+456*1024**2

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,j):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False));t.replace(p)
def rows(p):return [json.loads(s) for s in p.read_text().splitlines(keepends=True) if s.endswith('\n')]
def table():
    result={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid=,rss=,args='],text=True).splitlines():
        t=line.strip().split(None,3)
        if len(t)==4:result[int(t[0])]=(int(t[1]),int(t[2]),t[3])
    return result
def matching_jobs(source,output):
    return [pid for pid,v in table().items() if source in v[2] and str(output) in v[2] and str(R) in v[2]]
def snapshot():
    t=table();owned={pid for pid,v in t.items() if str(R) in v[2] and ('stage3_' in v[2] or 'isaacsim' in v[2])}
    while True:
        old=len(owned);owned.update(pid for pid,v in t.items() if v[0] in owned)
        if len(owned)==old:break
    rss=sum(t[pid][1] for pid in owned if pid in t)*1024;gpu=0
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10).splitlines():
        pid,mem=[s.strip() for s in line.split(',')]
        if int(pid) in owned:
            assert mem.isdigit(),'owned GPU accounting unavailable';gpu+=int(mem)*1024**2
    return {'owned_gpu_bytes':gpu,'owned_rss_bytes':rss,'free_bytes':shutil.disk_usage('/data').free}

def main():
    W.mkdir(exist_ok=False);state={'status':'waiting_original_sources','pid':os.getpid(),'formal_training_started':False}
    children=[];began=time.monotonic()
    def save(**kwargs):
        state.update(kwargs,elapsed_seconds=time.monotonic()-began);atomic(W/'status.json',state)
    def guard():
        s=snapshot();save(resources=s)
        assert s['free_bytes']>=FLOOR,'data disk reserve'
        assert s['owned_gpu_bytes']<=12*1024**3 and s['owned_rss_bytes']<=32*1024**3,'owned job resource protection'
    def run(command,env,cwd,logname):
        log=(W/logname).open('x')
        child=subprocess.Popen(command,env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child,log
    def wait(child,log):
        while child.poll() is None:
            guard();time.sleep(3)
        code=child.wait();log.close();assert code==0,'child failed; inspect preserved log'
    def audit(directory,count,name,env):
        output=W/(name+'.json');child,log=run([P,'-u',str(S/'src/stage3_audit_pick_batch.py'),'--input',str(directory),
            '--output',str(output),'--expected-count',str(count)],env,str(S),name+'.log')
        wait(child,log);report=json.loads(output.read_text());assert report['passed'] and report['candidates']==count
        return {'path':str(output),'sha256':sha(output)}
    lock=(O/'replacement_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        spec=json.loads((D/'env_spec.json').read_text());assert spec['spec_hash']=='74af067a'
        sample_env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
            'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':':'.join(spec['spec']['module_paths'])}
        sample_env.pop('CUDA_VISIBLE_DEVICES',None)
        old_sample=O/'candidate_remaining496_v1';old_opt=O/'postopt_remaining488_jit_v3'
        save()
        while True:
            guard()
            if all((p/'report.json').exists() for p in [old_sample,old_opt]):
                if not matching_jobs('stage3_sample_pick29_batch.py',old_sample) and not matching_jobs('stage3_pick500_postopt_follow_v3.py',old_opt):break
            if time.monotonic()-began>14400:raise TimeoutError('bounded wait for original jobs')
            time.sleep(15)
        assert json.loads((old_sample/'report.json').read_text())['generated_candidates']==496
        assert json.loads((old_opt/'report.json').read_text())['status']=='postopt_completed_native_pending'
        assert len(rows(old_opt/'records.jsonl'))==488
        save(status='auditing_completed_original_candidates')
        old_audit=audit(old_sample,496,'remaining496_candidate_audit',sample_env)
        initial=rows(O/'candidate_pilot_v1/records.jsonl')+rows(old_sample/'records.jsonl')
        assert sorted(r['candidate_id'] for r in initial)==list(range(500))
        for key in ['initial_noise_sha256','feature_content_sha256']:
            assert len({r[key] for r in initial})==500,'cross-batch candidate duplicate'
        atomic(W/'original500_cross_audit.json',{'passed':True,'candidate_count':500,'new_reference_admission_claimed':False,
            'remaining496_audit':old_audit,'pilot_records_sha256':sha(O/'candidate_pilot_v1/records.jsonl'),
            'remaining_records_sha256':sha(old_sample/'records.jsonl'),'ID_noise_feature_content_duplicates':0})
        new_sample=O/'candidate_replacement128_v1';new_opt=O/'postopt_replacement128_v1'
        assert not new_sample.exists() and not new_opt.exists(),'preserved outputs exist; never relaunch blindly'
        assert not matching_jobs('stage3_sample_pick29_batch.py',old_sample)
        sample_command=['timeout','--kill-after=20s','28800',P,'-u',str(S/'src/stage3_sample_pick29_batch.py'),
            '--output-dir',str(new_sample),'--count','128','--start-index','500','--batch-size','2']
        cfg=json.loads((D/'config.json').read_text());sample,sl=run(sample_command,sample_env,cfg['trajgen'],'replacement_sampler.log')
        tree=ast.parse((S/'src/stage3_postopt_queue.py').read_text())
        assignment=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='env' for n in x.targets))
        ns={'R':R,'S':S,'Path':Path,'os':os};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'existing verified env only','exec'),ns)
        opt_env=ns['env'];opt_env['CUDA_VISIBLE_DEVICES']='1';opt_env['PYTHONDONTWRITEBYTECODE']='1'
        opt_command=['timeout','--kill-after=20s','28800',P,'-u',str(S/'src/stage3_pick500_postopt_follow_v3.py'),
            '--records',str(new_sample/'records.jsonl'),'--output-dir',str(new_opt),
            '--initial-state',str(S/'reports/pick_table_bounded_0_v1/robot29.npz'),'--max-candidates','128','--follow']
        optimizer,ol=run(opt_command,opt_env,str(S),'replacement_optimizer.log')
        registry_path=O/'native_pipeline_registry.json';registry=json.loads(registry_path.read_text())
        registry['active_jobs'].extend([{'pid':sample.pid,'script':'stage3_sample_pick29_batch.py'},
            {'pid':optimizer.pid,'script':'stage3_pick500_postopt_follow_v3.py'}]);atomic(registry_path,registry)
        receipt={'sampler_pid':sample.pid,'optimizer_pid':optimizer.pid,'sample_command':sample_command,'opt_command':opt_command,
            'candidate_ids':[500,627],'count':128,'prior_source_jobs_ended':True,'original500_cross_audit_sha256':sha(W/'original500_cross_audit.json'),
            'source_hashes':{p.name:sha(p) for p in [S/'src/stage3_sample_pick29_batch.py',S/'src/stage3_pick500_postopt_follow_v3.py']},
            'note':'replacements not admitted references; no formal PPO or SuperMimic launch'}
        (W/'replacement_launch.json').open('x').write(json.dumps(receipt,indent=2));save(status='replacement_sources_running',replacement_launch=receipt)
        while sample.poll() is None or optimizer.poll() is None:
            guard()
            if sample.poll() is not None:assert sample.returncode==0,'replacement sampling failed'
            if optimizer.poll() is not None:assert optimizer.returncode==0,'replacement optimization failed'
            assert time.monotonic()-began<18000,'replacement queue time bound'
            time.sleep(5)
        wait(sample,sl);wait(optimizer,ol)
        assert len(rows(new_sample/'records.jsonl'))==128 and len(rows(new_opt/'records.jsonl'))==128
        save(status='auditing_replacement_candidates')
        new_audit=audit(new_sample,128,'replacement128_candidate_audit',sample_env)
        all_rows=initial+rows(new_sample/'records.jsonl');assert sorted(r['candidate_id'] for r in all_rows)==list(range(628))
        for key in ['initial_noise_sha256','feature_content_sha256']:
            assert len({r[key] for r in all_rows})==628,'cross-batch replacement duplicate'
        final={'status':'replacement128_completed_candidate_audited_native_admission_separate','candidate_count':628,
            'replacement_audit':new_audit,'original_audit':old_audit,'unique_IDs_noise_features':True,
            'replacement_optimizer_records_sha256':sha(new_opt/'records.jsonl'),'native_library_progress':str(O/'library_progress.json'),
            'not_a_final500_library_receipt':True,'formal_training_started':False}
        (W/'report.json').open('x').write(json.dumps(final,indent=2));save(**final)
    except BaseException as e:
        save(status='failed_preserved',error=repr(e))
        # Only this queue's newly launched groups may be stopped. Original jobs
        # and native admission are deliberately outside this signal list.
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:lock.close()

if __name__=='__main__':main()
