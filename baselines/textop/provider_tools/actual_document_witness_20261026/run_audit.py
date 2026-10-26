import pathlib,sys,json,hashlib,traceback
W=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('C:/Users/Pende/Documents/ChatGPT/机器人格斗动作捕捉数据生成/work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop')
sys.path.insert(0,str(B/'provider_tools'))
from session import remote
execution=json.loads((W/'document_execution.json').read_text(encoding='utf-8'))
assert execution['actual_ssh_exit']==0 and execution['local_ssh_reaped'] is True
try:
    raw=remote((W/'audit_remote.py').read_text(encoding='utf-8'),timeout=180)
    (W/'document_audit_raw.json').write_text(raw,encoding='utf-8')
    j=json.loads(raw);assert j['status']=='DOCUMENT_ACTUAL_AUDIT_PASSED'
    before=json.loads((W/'preflight_v2.json').read_text(encoding='utf-8'))
    assert j['source_sha256']==before['source_sha256'] and j['spec_sha256']==before['spec_sha256']
    assert j['teacher_artifact_sha256']==before['teacher_artifact_sha256']
    for rel,h in j['source_sha256'].items():assert hashlib.sha256((B/rel).read_bytes()).hexdigest()==h,rel
    j['pre_post_source_spec_teacher_artifacts_unchanged']=True
    (W/'document_audit.json').write_text(json.dumps(j,indent=2,ensure_ascii=False),encoding='utf-8')
    for key in ['receipt','exit','process_exit']:
        (W/('document_'+key+'.json')).write_text(json.dumps(j['document'][key],indent=2,ensure_ascii=False),encoding='utf-8')
    r,e=j['document']['receipt'],j['document']['exit']
    summary={'status':j['status'],'run':'quick_pilot_smoke_document_v1','actual_document_command':execution['documented_command'],'actual_ssh_exit':execution['actual_ssh_exit'],'actual_evaluator_exit':e['child_exit_code'],'supervision_passed':e['supervision_passed'],'closed_trials':r['finished_trials'],'natural_completions':r['complete_trials'],'failed_trials':r['failed_trials'],'control_calls':r['actual_vectorized_control_calls'],'scene_physics_steps':r['actual_scene_physics_steps'],'writers_closed':r['output_writers_closed'],'native_close_returned':r['native_close_returned'],'teardown_fallback_used':r['teardown_fallback_used'],'hashes':j['document_hashes'],'private_payload_audit':j['private_payload_audit'],'common_lock_available_after_exit':j['common_lock_available'],'sdk_conflicts_after_exit':j['sdk_conflicts'],'pre_post_source_spec_teacher_artifacts_unchanged':True,'documented_invocation_divergence':None,'full_failure_branch_exercised':r['failed_trials']>0,'limitations':['Early5000 teacher and eight shortest complete originals only; not full-library filtering or paper-complete quality','Native SDK close did not return; bounded fallback distinctly recorded after writers closed','All8 successful: actual failure branch not exercised','Consumed targets0..T-2; separate final-frameT-1 endpoint comparison not performed'],'preflight_v1_preserved':True,'shared_gpu_authorization_applied_in_v2':True,'public_shared_results_written':False}
    (W/'document_verification.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
except BaseException:
    (W/'audit_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
