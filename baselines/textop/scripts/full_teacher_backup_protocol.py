"""Provenance gate for NEW resumed-run snapshots; no Torch, signals or mutation."""
import hashlib,json,pathlib,re,time

RUN='filter_policy_paper69_resume5000_to40000_20261026_v1'
def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def decide(spec,receipt,completed,exits=None):
    assert spec['run']==RUN and completed in range(10000,40001,5000),'Only new resumed-run schedule'
    assert receipt['mode']=='formal' and receipt['status'] in ('RUNNING_FORMAL_FILTER_TRAINING','TEXTOP_FORMAL_FILTER_TRAINING_FINISHED')
    for key in ['plan_sha256','source_sha256','runtime_sha256','reference_fingerprint']:
        assert receipt[key]==spec['training_receipt_bindings'][key],key
    assert receipt['num_envs']==8192 and receipt['reference_sources']==8278
    assert receipt['resume_exact_state_verified'] is True
    count=receipt['completed_learning_iterations'];updates=receipt['optimizer_updates']
    assert type(count) is int and type(updates) is int and 5000<=count<=40000 and updates==20*count
    meta=receipt.get('latest_checkpoint')
    if meta is None:return {'ready':False,'status':'PENDING_NO_NEW_CHECKPOINT','observed_completed':count}
    actual=meta['completed_learning_iterations']
    assert actual in range(10000,40001,5000) and actual<=count and meta['actual_optimizer_updates']==20*actual
    assert meta['name']=='completed_{}.pt'.format(actual) and type(meta['bytes']) is int and 0<meta['bytes']<=128*1024**2
    assert re.fullmatch('[0-9a-f]{64}',meta['sha256'])
    assert actual<=completed,'A newer checkpoint exists; never weaken latest-only provenance'
    if actual<completed:return {'ready':False,'status':'PENDING_REQUESTED_CHECKPOINT','observed_completed':count,'latest_completed':actual}
    if completed==40000:
        if receipt['status']!='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED' or not receipt['output_writers_closed']:
            return {'ready':False,'status':'PENDING_FINAL_TRAINING_CLOSE','observed_completed':count}
        if not exits or any(exits.get(k) is None for k in ['process','supervisor','outer']):
            return {'ready':False,'status':'PENDING_ACTUAL_FINAL_EXITS','observed_completed':count}
        for key in ['process','supervisor']:
            e=exits[key];assert e['child_exit_code']==0 and e['child_reaped'] and e['supervision_passed'] and not e['timed_out'] and e.get('supervisor_error') is None
        assert exits['outer']['actual_returncode']==0
    return {'ready':True,'status':'METADATA_READY_FILE_CHECKS_REQUIRED','observed_completed':count,'metadata':meta}

def preflight(root,spec,completed,now=None):
    root=pathlib.Path(root);sp=root/'configs/full-teacher-resume-spec.json'
    assert sha(sp)==spec['resume_spec_sha256']
    resume=json.loads(sp.read_text());assert resume['run']==spec['run']==RUN
    for name,h in resume['frozen_bindings'].items():assert sha(root/name)==h,name
    assert sha(root/'scripts/supervise_full_teacher_resume.py')==resume['source_sha256']
    for name,h in spec['bound_backup_files'].items():assert sha(root/name)==h,name
    out=root/'tracker_validation'/RUN;rp=out/'receipt.json';raw=rp.read_bytes();receipt=json.loads(raw)
    prefix='full_resume_'+RUN;exits={}
    for key,suffix in [('process','_process_exit.json'),('supervisor','_exit.json'),('outer','_outer_exit.json')]:
        p=root/'receipts'/(prefix+suffix);exits[key]=json.loads(p.read_text()) if p.exists() else None
    decision=decide(spec,receipt,completed,exits)
    decision.update(run=RUN,requested_completed=completed,training_receipt_sha256=hashlib.sha256(raw).hexdigest(),
                    sdk_dispatches=0,checkpoint_loads=0,export_files_created=0)
    if not decision['ready']:return decision
    cp=out/decision['metadata']['name'];assert cp.resolve()==cp and not cp.is_symlink() and cp.stat().st_uid==1013
    before=cp.stat();age=(time.time() if now is None else now)-before.st_mtime
    if age<=30:return dict(decision,ready=False,status='PENDING_FILE_STABILITY',file_age_seconds=age)
    assert before.st_size==decision['metadata']['bytes'] and sha(cp)==decision['metadata']['sha256']
    after=cp.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
    latest=json.loads(rp.read_text());again=decide(spec,latest,completed,exits)
    assert again['ready'] and again['metadata']==decision['metadata'],'Source receipt changed while hashing'
    return dict(decision,status='RESUMED_CHECKPOINT_READY_FOR_CPU_AUDIT',checkpoint=str(cp),file_age_seconds=age)
