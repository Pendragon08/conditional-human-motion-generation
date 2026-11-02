"""Synthetic metadata rejection tests only; never import Torch or send signals."""
import copy,json,pathlib,sys
from full_teacher_backup_protocol import RUN,decide
from teacher_evaluation_control import validate_backup,validate_identity,ROOT
def main():
    root=pathlib.Path(__file__).resolve().parents[1];s=json.loads((root/'configs/full-teacher-backup-spec.json').read_text());control=json.loads((root/'configs/teacher10000-evaluation-spec.json').read_text())
    r=dict(s['training_receipt_bindings'],mode='formal',status='RUNNING_FORMAL_FILTER_TRAINING',num_envs=8192,reference_sources=8278,resume_exact_state_verified=True,completed_learning_iterations=10000,optimizer_updates=200000,
      latest_checkpoint=dict(name='completed_10000.pt',completed_learning_iterations=10000,actual_optimizer_updates=200000,bytes=90799598,sha256='a'*64))
    checks=[]
    def test(name,fn,allowed):
        try:fn();ok=True
        except (AssertionError,KeyError,TypeError,ValueError):ok=False
        assert ok==allowed,name;checks.append(name)
    test('valid_synthetic_10000',lambda:assert_ready(decide(s,r,10000)),True)
    pending=copy.deepcopy(r);pending.pop('latest_checkpoint');assert not decide(s,pending,10000)['ready'];checks.append('absent_checkpoint_pending')
    for key,val in [('num_envs',1),('reference_sources',64),('resume_exact_state_verified',False),('optimizer_updates',199999),('plan_sha256','bad'),('source_sha256','bad'),('runtime_sha256','bad'),('reference_fingerprint','bad')]:
        rr=dict(r);rr[key]=val;test('reject_'+key,lambda rr=rr:decide(s,rr,10000),False)
    for key,val in [('name','../completed_10000.pt'),('bytes',129*1024**2),('sha256','bad'),('actual_optimizer_updates',100000)]:
        rr=copy.deepcopy(r);rr['latest_checkpoint'][key]=val;test('reject_metadata_'+key,lambda rr=rr:decide(s,rr,10000),False)
    test('reject_old5000_request',lambda:decide(s,r,5000),False)
    newer=copy.deepcopy(r);newer.update(completed_learning_iterations=15000,optimizer_updates=300000);newer['latest_checkpoint'].update(name='completed_15000.pt',completed_learning_iterations=15000,actual_optimizer_updates=300000)
    test('reject_newer_latest',lambda:decide(s,newer,10000),False)
    last=copy.deepcopy(r);last.update(completed_learning_iterations=40000,optimizer_updates=800000,output_writers_closed=False);last['latest_checkpoint'].update(name='completed_40000.pt',completed_learning_iterations=40000,actual_optimizer_updates=800000)
    assert not decide(s,last,40000)['ready'];checks.append('final_close_pending')
    last.update(status='TEXTOP_FORMAL_FILTER_TRAINING_FINISHED',output_writers_closed=True);assert not decide(s,last,40000)['ready'];checks.append('final_exits_pending')
    ex=dict(child_exit_code=0,child_reaped=True,supervision_passed=True,timed_out=False,supervisor_error=None);exits=dict(process=ex,supervisor=ex,outer={'actual_returncode':0})
    test('valid_synthetic_final_exits',lambda:assert_ready(decide(s,last,40000,exits)),True)
    bad=copy.deepcopy(exits);bad['outer']['actual_returncode']=1;test('reject_final_bad_exit',lambda:decide(s,last,40000,bad),False)
    cp=r['latest_checkpoint'];b={'status':'FORMAL_CHECKPOINT_PRIVATE_FULL_AND_GITHUB_INFERENCE_VERIFIED','run':RUN,'completed':10000,
       'private_full_two_location_sha_verified':True,'inference_three_location_sha_verified':True,'original_server_checkpoint_retained':True,'data_deleted':False,'full_checkpoint_publicly_uploaded':False,
       'adapter':{'spec_sha256':control['backup_spec_sha256'],'run':RUN},'github_digest':'sha256:'+'b'*64,
       'audit':{'run':RUN,'completed_learning_iterations':10000,'actual_optimizer_updates':200000,'optimizer_step_min':200000,'optimizer_step_max':200000,
          'full_checkpoint':{'path':str(ROOT/'tracker_validation'/RUN/'completed_10000.pt'),'bytes':cp['bytes'],'sha256':cp['sha256']},'public_inference':{'sha256':'b'*64}}}
    test('valid_synthetic_backup',lambda:validate_backup(control,b,cp),True)
    for key,val in [('run','oldrun'),('private_full_two_location_sha_verified',False),('github_digest','bad'),('data_deleted',True),('full_checkpoint_publicly_uploaded',True)]:
        bb=copy.deepcopy(b);bb[key]=val;test('reject_backup_'+key,lambda bb=bb:validate_backup(control,bb,cp),False)
    ident=control['process_identity'];trainer=str(ROOT/'scripts/train_filter_policy.py');out=str(ROOT/'tracker_validation'/RUN)
    cmd=[trainer,'--output',out,'--mode','formal','--device','cuda:1','--resume','trusted']
    launch={'run':RUN,'spec_sha256':control['resume_spec_sha256'],'command':cmd};live={'child_pid':ident['group']}
    member=dict(ident,uid=1013,session=ident['group'],python_executable=True,argv=cmd)
    test('valid_synthetic_identity',lambda:validate_identity(control,launch,live,[member]),True)
    for key,val in [('start_ticks',-1),('uid',0),('session',-1),('python_executable',False)]:
        mm=dict(member);mm[key]=val;test('reject_identity_'+key,lambda mm=mm:validate_identity(control,launch,live,[mm]),False)
    test('reject_two_sdk_processes',lambda:validate_identity(control,launch,live,[member,member]),False)
    print(json.dumps({'status':'TEACHER10000_METADATA_TESTS_PASSED','checks':checks,'count':len(checks),'synthetic_metadata_only':True,'signals_sent':0,'sdk_runs':0,'checkpoint_loads':0,'training_quality_claimed':False}))
def assert_ready(x):assert x['ready']
if __name__=='__main__':main()
