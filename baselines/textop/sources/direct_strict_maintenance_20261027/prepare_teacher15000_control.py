"""Prepare a new user-authorized15k control; never execute stop or alter training."""
import hashlib,json,pathlib,sys,base64
W=pathlib.Path(__file__).resolve().parents[2]
B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sys.path.insert(0,str(B/'provider_tools'));from session import remote
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
original=(B/'scripts/teacher_evaluation_control.py').read_text(encoding='utf-8')
code=original.replace('10000','15000').replace('200000','300000')
code=code.replace('from full_teacher_backup_protocol import','from filter_recovery_backup_protocol import')
code=code.replace('teacher15000-evaluation-spec.json','teacher15000-direct-evaluation-spec.json')
code=code.replace('configs/full-teacher-backup-spec.json','configs/filter-recovery-backup-spec.json')
code=code.replace('configs/full-teacher-resume-spec.json','configs/teacher10000-filter-recovery-spec.json')
code=code.replace('scripts/supervise_full_teacher_resume.py','scripts/supervise_teacher10000_filter_recovery.py')
code=code.replace("prefix='full_resume_'+RUN","prefix='filter_recovery_'+RUN")
code=code.replace('User-authorized 10k evaluation boundary','User-authorized15k direct full-library physics boundary')
target=B/'scripts/teacher15000_evaluation_control.py';assert not target.exists();target.write_text(code,encoding='utf-8',newline='\n')
spec=dict(schema='textop.teacher15000_direct_evaluation.v1',run='filter_policy_paper69_resume10000_filter_recovery_20261027_v1',
 user_request='直接进行下一步物理筛选，不要再训练跟踪器了。',handoff_enabled=True,evaluation_checkpoint=15000,paper_reference_iterations=40000,
 process_identity=dict(pid=4190451,start_ticks=1427248665,group=4190443),
 resume_spec_sha256=sha(B/'configs/teacher10000-filter-recovery-spec.json'),backup_spec_sha256=sha(B/'configs/filter-recovery-backup-spec.json'),
 bound_control_sources={n:sha(B/n) for n in ['scripts/teacher15000_evaluation_control.py','scripts/quick_pidfd.py','scripts/filter_recovery_backup_protocol.py']},
 direct_strict_source_count=8278,keep_original_ema_gate_result=True,continue_teacher_authorized=False,evaluation_launch_enabled=False)
sp=B/'configs/teacher15000-direct-evaluation-spec.json';assert not sp.exists();sp.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
paths=[target,sp]
payload={p.relative_to(B).as_posix():base64.b64encode(p.read_bytes()).decode() for p in paths}
script="import pathlib,base64,hashlib,json,os\nr=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');assert os.getuid()==1013\ndata="+repr(payload)+"\nresult={}\nfor n,v in data.items():\n p=r/n;assert not p.exists();raw=base64.b64decode(v)\n with p.open('xb') as f:f.write(raw)\n result[n]=hashlib.sha256(p.read_bytes()).hexdigest()\nprint(json.dumps(result))\n"
result=json.loads(remote(script));assert result=={p.relative_to(B).as_posix():sha(p) for p in paths}
print(json.dumps(result))
