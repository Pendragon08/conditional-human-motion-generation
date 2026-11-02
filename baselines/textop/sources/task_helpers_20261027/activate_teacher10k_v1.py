import base64,hashlib,json,pathlib,sys,shutil
B=pathlib.Path('work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop');sys.path.insert(0,str(B/'provider_tools'));from session import remote
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();R=B/'results/teacher10000_evaluation_20261026'
doc=pathlib.Path('work/textop_reproduction_20261024/teacher10000_document_v1');j=json.loads((doc/'SUMMARY.json').read_text(encoding='utf-8'))
assert j['documented_invocation_count']==1 and j['remote_control_actual_exit']==0 and not j['actionable_findings'] and not j['doc_reality_divergences']
assert j['verified_source_or_spec_hash_count']==28 and j['checkpoint_loads']==0 and not j['signals_sent']
target=R/'independent_document_v1';target.mkdir(exist_ok=False)
for n in ['REPORT.md','SUMMARY.json','source_manifest.json','invocation_execution.json','supplemental_readonly_evidence.json']:
 if (doc/n).exists():shutil.copy2(doc/n,target/n)
sp=B/'configs/teacher10000-evaluation-spec.json';old=sp.read_bytes();s=json.loads(old);assert not s['handoff_enabled']
s['handoff_enabled']=True;s['independent_document']={'summary_sha256':sha(doc/'SUMMARY.json'),'report_sha256':sha(doc/'REPORT.md'),'actual_exit':0,'actual_checkpoint_audit_and_stop_pending':True}
new=(json.dumps(s,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
result=json.loads(remote('''import pathlib,json,hashlib,base64,subprocess,sys
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');p=r/'configs/teacher10000-evaluation-spec.json';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(p)==%r
for n in ['teacher10000_stop_requested.json','teacher10000_controlled_stop.json','teacher10000_backup_verified.json']:assert not (r/'receipts'/n).exists()
old=json.loads(p.read_text());before=subprocess.run([sys.executable,str(r/'scripts/teacher_evaluation_control.py'),'--action','observe'],capture_output=True,text=True,timeout=45)
assert before.returncode==0,(before.stdout,before.stderr);observed=json.loads(before.stdout);assert observed['observed_completed']<10000 and not observed['signals_sent']
data=base64.b64decode(%r);new=json.loads(data);expected=dict(old);expected['handoff_enabled']=True;expected['independent_document']=new['independent_document'];assert expected==new
q=p.with_suffix('.activation_partial');assert not q.exists()
with q.open('xb') as f:f.write(data)
q.replace(p)
after=subprocess.run([sys.executable,str(r/'scripts/teacher_evaluation_control.py'),'--action','observe'],capture_output=True,text=True,timeout=45)
assert after.returncode==0,(after.stdout,after.stderr)
print(json.dumps({'new_spec_sha256':sha(p),'observation':json.loads(after.stdout),'training_signals':0,'sdk_runs':0,'active_training_files_unchanged':True}))
'''%(hashlib.sha256(old).hexdigest(),base64.b64encode(new).decode()),timeout=110))
sp.write_bytes(new);assert sha(sp)==result['new_spec_sha256']
(R/'activation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
names=['configs/teacher10000-evaluation-spec.json','configs/full-teacher-backup-spec.json','scripts/full_teacher_backup_protocol.py','scripts/inspect_full_teacher_checkpoint.py','scripts/teacher_evaluation_control.py','provider_tools/backup_full_teacher_checkpoint.py','provider_tools/teacher10000_coordinator.py']
(R/'enabled_runtime_bytes_v1.json').write_text(json.dumps({'before_control_spec_base64':base64.b64encode(old).decode(),'files':{n:{'sha256':sha(B/n),'base64':base64.b64encode((B/n).read_bytes()).decode()} for n in names}},indent=2),encoding='utf-8')
print(json.dumps({'activated':True,'state':result['observation']['status'],'spec_sha256':sha(sp),'no_signal_sent':True}))
