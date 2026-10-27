import ast,base64,hashlib,json,pathlib,subprocess,sys
B=pathlib.Path('work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop');sys.path.insert(0,str(B/'provider_tools'))
from session import remote
ROOT='/data/code/ouyanghao/oyh/textop-reproduction-20261024';R=B/'results/teacher10000_evaluation_20261026'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
identity=json.loads(remote('''import json,pathlib,os
r=pathlib.Path(%r);run='filter_policy_paper69_resume5000_to40000_20261026_v1'
live=json.loads((r/'receipts'/('full_resume_'+run+'_live.json')).read_text());members=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  pid=int(p.name)
  if p.stat().st_uid!=1013 or os.getpgid(pid)!=live['child_pid']:continue
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\\0')
  if (p/'exe').resolve().name.startswith('python') and str(r/'scripts/train_filter_policy.py') in argv:
   members.append({'pid':pid,'start_ticks':int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),'group':os.getpgid(pid)})
 except (OSError,ProcessLookupError):pass
assert len(members)==1
print(json.dumps(members[0]))
'''%ROOT,timeout=40))
plan={'schema':'textop.teacher10000_evaluation.v1','run':'filter_policy_paper69_resume5000_to40000_20261026_v1',
 'user_request':'2026-10-26: evaluate physical filtering at10000 iterations first; improve motion semantics; notify when paper-level complete reproduction is achieved.',
 'handoff_enabled':False,'evaluation_checkpoint':10000,'paper_reference_iterations':40000,'process_identity':identity,
 'resume_spec_sha256':sha(B/'configs/full-teacher-resume-spec.json'),'backup_spec_sha256':sha(B/'configs/full-teacher-backup-spec.json'),
 'bound_control_sources':{n:sha(B/n) for n in ['scripts/teacher_evaluation_control.py','scripts/quick_pidfd.py','scripts/full_teacher_backup_protocol.py']},
 'evaluation_launch_enabled':False,'evaluation_entrypoint':None,
 'evaluation_protocol':{'first_comparison':'Same previously fixed64 source IDs, all42 successes and22 failures retained; same full-duration strict thresholds and contact, compare5k vs10k without selecting best retries.',
   'additional_coverage':'After executor author/fresh validation, fixed held-out representative sources including longer durations and missing lexical groups; private selection frozen before outcomes.',
   'metrics':['complete-duration success/failure/unattempted','root/body/joint tracking error','feet slip/contact','full reference endpoint check','per-source and duration-stratum paired changes'],
   'weak_teacher_failure_is_not_absolute_motion_infeasibility':True,'whole_library_acceptance':False,'continue_to40000_if_needed':True},
 'semantics':{'quality_accepted':False,'causal_reason_established':False,'do_not_modify_active_training':True,
   'controlled_comparisons':['Same seed/history different prompts and unconditional branch','Same prompt different seeds','Fixed held-out text families and original full5-step CFG5 sampling','VAE posterior reconstruction versus autoregressive generation','Endpoint and semantic motion measurements plus visual review'],
   'improvements':'Use measured label coverage and held-out errors to expand physically accepted, source-separated training corpus and adjust only explicit experimental variants. Preserve paper baseline, negative results, data and every checkpoint.'},
 'notification':'Notify when paper complete reproduction quality and all required stages are actually accepted, and for material results/failures; not every healthy polling interval.'}
sp=B/'configs/teacher10000-evaluation-spec.json';assert not sp.exists();sp.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
files=['scripts/full_teacher_backup_protocol.py','scripts/inspect_full_teacher_checkpoint.py','configs/full-teacher-backup-spec.json','scripts/teacher_evaluation_control.py',
 'scripts/validate_teacher10000_protocol.py','scripts/audit_quick_text_coverage.py','configs/teacher10000-evaluation-spec.json']
for n in files:
 if n.endswith('.py'):ast.parse((B/n).read_text(encoding='utf-8'))
local=subprocess.run([sys.executable,str(B/'scripts/validate_teacher10000_protocol.py')],capture_output=True,text=True,encoding='utf-8')
(R/'metadata_cpu_local_v1.json').write_text(json.dumps(dict(exit_code=local.returncode,stdout=local.stdout,stderr=local.stderr),indent=2),encoding='utf-8');assert local.returncode==0,local.stderr
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in files}
code='''import pathlib,base64,json,hashlib,subprocess,sys,os
r=pathlib.Path(%r);payload=%r;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
resume=json.loads((r/'configs/full-teacher-resume-spec.json').read_text());frozen=dict(resume['frozen_bindings']);frozen['scripts/supervise_full_teacher_resume.py']=resume['source_sha256']
assert all(sha(r/n)==h for n,h in frozen.items())
for n,v in payload.items():
 p=r/n;raw=base64.b64decode(v)
 if p.exists():assert p.read_bytes()==raw,n
 else:
  with p.open('xb') as f:f.write(raw)
v=subprocess.run([sys.executable,str(r/'scripts/validate_teacher10000_protocol.py')],capture_output=True,text=True,timeout=30)
assert v.returncode==0,(v.stdout,v.stderr)
o=subprocess.run([sys.executable,str(r/'scripts/teacher_evaluation_control.py'),'--action','observe'],capture_output=True,text=True,timeout=45)
assert o.returncode==0,(o.stdout,o.stderr)
assert all(sha(r/n)==h for n,h in frozen.items())
print(json.dumps({'cpu':{'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr},'observe':json.loads(o.stdout),'active_frozen_files_unchanged':True,'new_files':{n:sha(r/n) for n in payload}}))
'''%(ROOT,payload)
v=json.loads(remote(code,timeout=90));(R/'staging_v1.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
(R/'runtime_bytes_disabled_v1.json').write_text(json.dumps({'files':{n:{'sha256':sha(B/n),'base64':payload[n]} for n in files}},indent=2),encoding='utf-8')
print(json.dumps({'local':local.returncode,'server':v['cpu']['exit_code'],'observe':v['observe']['status'],'counts':json.loads(v['cpu']['stdout'])['count']}))
