"""Create disabled actual-pass-only corpus entry; never execute any job."""
import pathlib,json,ast,hashlib
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(rel,text):
 p=B/rel;assert not p.exists();p.write_text(text,encoding='utf-8');ast.parse(text) if p.suffix=='.py' else None
s=(B/'scripts/build_generator8278_corpus.py').read_text(encoding='utf-8')
s=s.replace('user-authorized PRIVATE full pre-filter generator corpus','PRIVATE actual-strict-pass15000 generator corpus').replace('No optimizer or model training here. This corpus is valid only for the explicitly\nauthorized pre-filter experiment; it is never paper-filtered acceptance.','No optimizer or model training here. Real closed8278 replay/audit proof is required.\nThis user-directed route bypassed the historical EMA gate and is recorded as such.')
s=s.replace('textop.generator8278_corpus.v1','textop.filtered15000_corpus.v1')
s=s.replace("spec=json.loads(pathlib.Path(a.spec).read_text());", "spec=json.loads(pathlib.Path(a.spec).read_text());")
key="assert sha(spec['preprocessing_receipt'])==spec['preprocessing_receipt_sha256']"
s=s.replace(key,"from filtered15000_inputs import load_selection,select_prepared_rows\n strict_audit,selected_sources=load_selection(spec);selected_count=len(selected_sources)\n "+key)
s=s.replace("'candidate_view_only':True","'candidate_view_only':False,'input_sources_strict_filtered':True")
key="assert {s:sum(x['split']==s for x in rows) for s in ['train','val']}=={'train':6203,'val':2075}"
s=s.replace(key,key+"\n  rows=select_prepared_rows(rows,selected_sources)\n  assert len(rows)==selected_count and all(any(x['split']==s for x in rows) for s in ['train','val'])")
s=s.replace("'physical_filter_passed':False","'physical_filter_passed':True")
s=s.replace("'candidate_only':True","'candidate_only':False,'actual_strict_input_passed':True")
s=s.replace('PREFILTER_8278_CORPUS_BUILT_NOT_PHYSICALLY_FILTERED','DIRECT_STRICT15000_CORPUS_BUILT_LOADER_PENDING')
s=s.replace("'source_sequences':len(rows),'motion_variants'", "'source_sequences':len(rows),'input_library_sources':8278,'strict_audit_receipt_sha256':spec['strict_audit_receipt_sha256'],'paper_two_stage_route_reproduced':False,'motion_variants'")
s=s.replace('USER_AUTHORIZED_PREFILTER_TRAINING_ONLY_REFIT_AFTER_ACTUAL_FILTERING','ACTUAL_STRICT15000_PASSES_TRAIN_ONLY_REFIT')
s=s.replace("meta['source_sequences']==spec['expected_sources']","meta['source_sequences']==selected_count")
s=s.replace('GENERATOR8278_CORPUS_LOADER_WITNESS_PASSED','FILTERED15000_CORPUS_LOADER_WITNESS_PASSED')
s=s.replace("'User-authorized8278-source pre-filter corpus; no optimizer/physical filtering/semantic acceptance. Paper-filtered dataset acceptance remains false'","'Actual standard strict passes at15000; original references plus BABEL. User route skipped historical EMA. No optimizer/semantic or paper two-stage acceptance.'")
s=s.replace("receipt['prefilter_corpus_loader_verified']=True;receipt['paper_filtered_dataset_accepted']=False","receipt['actual_strict_input_corpus_loader_verified']=True;receipt['paper_filtered_dataset_accepted']=False\n receipt['strict_audit_receipt_sha256']=spec['strict_audit_receipt_sha256'];receipt['input_library_sources']=8278")
s=s.replace("'8278-source pre-filter training explicitly selected by user2026-10-27'","'Actual direct strict15000 pass corpus; generation auto-authorized by user2026-10-27'")
put('scripts/build_filtered15000_corpus.py',s)
s=(B/'scripts/supervise_generator8278_corpus.py').read_text(encoding='utf-8')
s=s.replace('generator8278','filtered15000').replace('pre-filter','actual-pass-only')
key="assert sha(ROOT/'scripts/build_filtered15000_corpus.py')==s['builder_sha256'] and s['build_authorized']"
s=s.replace(key,key+"\nfrom filtered15000_inputs import load_selection\nload_selection(s)\nfor n,h in s['bound_sources'].items():assert sha(ROOT/n)==h,n")
s=s.replace('GENERATOR8278_CORPUS_LOADER_WITNESS_PASSED','FILTERED15000_CORPUS_LOADER_WITNESS_PASSED')
put('scripts/supervise_filtered15000_corpus.py',s)
s=json.loads((B/'configs/generator8278-corpus-spec.json').read_text(encoding='utf-8'))
s.update(schema='textop.filtered15000_corpus.v1',build_authorized=False,candidate_view='/dev/shm/ouyanghao_textop_20261025/filtered15000_actual_pass_v1',
 output_root='/data/code/ouyanghao/oyh/textop-reproduction-20261024/filtered15000_20261027/corpus_runs',
 strict_audit_receipt=None,strict_audit_receipt_sha256=None,strict_audit_supervisor_sha256=None,
 scope='Disabled until real full8278 closed strict audit. Actual passes only; original reference and BABEL; no historical EMA/paper two-stage claim.',
 statistics='Refit float64 TRAIN-only69D stats after actual strict source selection and exact-pose family quarantine',
 builder_sha256=sha(B/'scripts/build_filtered15000_corpus.py'),supervisor_sha256=sha(B/'scripts/supervise_filtered15000_corpus.py'))
s['bound_sources']['scripts/filtered15000_inputs.py']=sha(B/'scripts/filtered15000_inputs.py')
put('configs/filtered15000-corpus-spec.json',json.dumps(s,indent=2)+'\n')
audit=dict(schema='textop.closed_direct_strict15000_audit_spec.v1',launch_enabled=False,
 executor_sha256=sha(B/'scripts/audit_teacher15000_full_result.py'),supervisor_sha256=sha(B/'scripts/supervise_teacher15000_full_audit.py'),
 minimum_ram_free_bytes=16*1024**3,minimum_disk_free_bytes=3*1024**3,timeout_seconds=900,
 bound_sources={n:sha(B/n) for n in ['scripts/teacher15000_strict_protocol.py','scripts/strict_tracking_coverage.py','scripts/teacher15000_evaluation_control.py','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py']},
 actual_real_full_result_audited=False,filtered_corpus_accepted=False,optimizer_updates=0,new_gpu_contexts=0)
put('configs/teacher15000-full-result-audit-spec.json',json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(status='NEW_DISABLED_CPU_AUDIT_AND_FILTERED_CORPUS_SOURCE_PREPARED',actual_audit_runs=0,actual_corpus_runs=0,new_gpu_contexts=0)))
