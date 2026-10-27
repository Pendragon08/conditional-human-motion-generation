import pathlib,json,hashlib,ast
B=pathlib.Path('work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
obs=json.loads((B/'results/teacher10000_evaluation_20261026/initial_observation.json').read_text(encoding='utf-8'))
run=obs['resume']['run'];root='/data/code/ouyanghao/oyh/textop-reproduction-20261024'
spec={'schema':'textop.full_teacher_backup.v1','run':run,'resume_spec_sha256':sha(B/'configs/full-teacher-resume-spec.json'),
 'training_receipt_bindings':{k:obs['receipt'][k] for k in ['plan_sha256','source_sha256','runtime_sha256','reference_fingerprint']},
 'bound_backup_files':{n:sha(B/n) for n in ['scripts/full_teacher_backup_protocol.py','scripts/inspect_full_teacher_checkpoint.py','scripts/inspect_formal_filter_checkpoint_v2.py']},
 'audit_run':'resumed_author_v1','backup_enabled':True,'positive_resumed_checkpoint_audit_completed':False,
 'validation_scope':'New metadata gates; trusted original CPU serializer reused. Future actual snapshot still requires real load/export/exit and private backup.'}
(B/'configs/full-teacher-backup-spec.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
old=(B/'provider_tools/backup_formal_filter_checkpoint.py').read_text(encoding='utf-8')
new=old.replace("RUN = 'filter_policy_paper69_solid_floor_20261026_v1'",'RUN = '+repr(run)).replace("a.completed in [1000] + list(range(5000, 40001, 5000))","a.completed in range(10000, 40001, 5000)")
new=new.replace('results/formal_filter_checkpoints_20261026','results/full_teacher_checkpoints_20261026').replace("'formal_filter_round{}_backup.json'","'full_teacher_round{}_backup.json'")
start=new.index("        source = B / 'scripts/inspect_formal_filter_checkpoint_v2.py'")
end=new.index("        (folder / 'cpu_audit.json')",start)
prefix='''        spec = json.loads((B/'configs/full-teacher-backup-spec.json').read_text(encoding='utf-8'))
        expected_spec = sha(B/'configs/full-teacher-backup-spec.json')
        script = r\x27\x27\x27
import pathlib,os,json,hashlib,subprocess,sys
ROOT=pathlib.Path(ROOT_LITERAL);assert os.getuid()==1013;os.sched_setaffinity(0,{0,1,2,3})
sys.path.insert(0,str(ROOT/'scripts'))
from full_teacher_backup_protocol import preflight,sha
sp=ROOT/'configs/full-teacher-backup-spec.json';assert sha(sp)==EXPECTED_SPEC
s=json.loads(sp.read_text());assert s['run']==RUN_LITERAL and s['backup_enabled']
pre=preflight(ROOT,s,COMPLETED);assert pre['ready'],pre
out=ROOT/'checkpoint_backups'/('round_'+str(COMPLETED))/s['audit_run'];audit=out/'audit.json';adapter=out/'adapter_proof.json'
record=ROOT/'receipts'/('full_teacher_checkpoint_cpu_'+str(COMPLETED)+'_execution.json')
if not audit.exists():
 assert not out.exists() and not record.exists(),'Keep incomplete audit; inspect rather than retry'
 result=subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/inspect_full_teacher_checkpoint.py'),'--completed',str(COMPLETED)],capture_output=True,text=True,timeout=180,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 execution={'child_exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'sdk_runs':0}
 with record.open('x') as f:json.dump(execution,f,indent=2)
 assert result.returncode==0,(result.stderr,result.stdout)
else:execution=json.loads(record.read_text())
proof=json.loads(audit.read_text());ap=json.loads(adapter.read_text())
assert execution['child_exit_code']==0 and proof['run']==s['run'] and proof['completed_learning_iterations']==COMPLETED
assert proof['source_sha256']==s['bound_backup_files']['scripts/inspect_formal_filter_checkpoint_v2.py']
assert ap['spec_sha256']==EXPECTED_SPEC and ap['audit_sha256']==sha(audit)
assert ap['adapter_sha256']==s['bound_backup_files']['scripts/inspect_full_teacher_checkpoint.py']
assert ap['protocol_sha256']==s['bound_backup_files']['scripts/full_teacher_backup_protocol.py']
after=preflight(ROOT,s,COMPLETED);assert after['ready'] and after['metadata']==pre['metadata']
print(json.dumps({'audit':proof,'execution':execution,'adapter':ap,'new_run_provenance_verified':True,'preflight':after}))
\x27\x27\x27.replace('ROOT_LITERAL',repr(ROOT)).replace('RUN_LITERAL',repr(RUN)).replace('EXPECTED_SPEC',repr(expected_spec)).replace('COMPLETED',str(a.completed))
        audited = json.loads(remote(script, timeout=210))
'''
new=new[:start]+prefix+new[end:]
new=new.replace("state.update(status='CPU_AUDITED_DOWNLOAD_PENDING', audit=proof, data_deleted=False)","state.update(status='CPU_AUDITED_DOWNLOAD_PENDING', audit=proof, adapter=audited['adapter'], run=RUN, data_deleted=False)")
assert 'filter_policy_paper69_solid_floor_20261026_v1' not in new
ast.parse(new);(B/'provider_tools/backup_full_teacher_checkpoint.py').write_text(new,encoding='utf-8')
print(json.dumps({'backup_spec_sha256':sha(B/'configs/full-teacher-backup-spec.json'),'new_provider_created':True}))
