"""New CPU fixtures only; signals exactly one subprocess created by this test."""
import argparse,copy,hashlib,json,os,pathlib,signal,subprocess,sys,tempfile,time
B=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(B/'scripts'))
from ldm100k_control_protocol import checkpoint_ready,verify_checkpoint,identity,open_bound_pidfd,send_once,RESTORE_KEYS
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
 assert os.name=='posix','Run these fixtures on server Python3, never target a production PID'
 with tempfile.TemporaryDirectory(prefix='textop_ldm100k_cpu_') as tmp:
  root=pathlib.Path(tmp);cp=root/'ldm_state_100000.pth';cp.write_bytes(b'CPU fixture, not a model checkpoint')
  h=hashlib.sha256(cp.read_bytes()).hexdigest();ck=dict(path=str(cp),manager_step=100000,actual_optimizer_updates=100000,bytes=cp.stat().st_size,sha256=h)
  policy=dict(target_primitive_attempts=100000,checkpoint_path=str(cp),checkpoint_upper_bound_bytes=1024,original_spec_sha256='a'*64,
   original_entry_sha256='b'*64,original_runtime_binding_sha256='c'*64,corpus_receipt_sha256='d'*64,data_acceptance_sha256='e'*64,strict_audit_receipt_sha256='f'*64,vae_checkpoint_sha256='1'*64,vae_inference_sha256='2'*64)
  receipt=dict(mode='train',spec_sha256='a'*64,source_sha256='b'*64,runtime_binding_sha256='c'*64,
   corpus_receipt_sha256='d'*64,data_acceptance_sha256='e'*64,strict_audit_receipt_sha256='f'*64,
   vae_pretraining_attempts=100000,vae_checkpoint_sha256='1'*64,vae_inference_sha256='2'*64,full_sampling_history_calls=4,selected_rollout_history_primitives=2,completed_primitive_attempts=100000,optimizer_updates=100000,skipped_updates=0,checkpoints=[ck],latest_file_resume_verification={k:True for k in RESTORE_KEYS})
  assert checkpoint_ready(receipt,policy)==ck;verify_checkpoint(ck,root)
  zero=copy.deepcopy(receipt);zero['full_sampling_history_calls']=0;zero['selected_rollout_history_primitives']=0;assert checkpoint_ready(zero,policy)==ck
  pending=copy.deepcopy(receipt);pending['checkpoints']=[];assert checkpoint_ready(pending,policy) is None
  bad=[]
  for key,value in [('spec_sha256','0'*64),('source_sha256','0'*64),('runtime_binding_sha256','0'*64),('corpus_receipt_sha256','0'*64),('vae_checkpoint_sha256','0'*64),('vae_inference_sha256','0'*64)]:
   q=copy.deepcopy(receipt);q[key]=value;bad.append(q)
  q=copy.deepcopy(receipt);q['latest_file_resume_verification']['rng_and_generator_restored_exactly']=False;bad.append(q)
  q=copy.deepcopy(receipt);q['checkpoints'][0]['path']=str(root/'other.pth');bad.append(q)
  for q in bad:
   try:checkpoint_ready(q,policy)
   except AssertionError:pass
   else:raise AssertionError('Invalid fixture accepted')
  corrupted=dict(ck,sha256='0'*64)
  try:verify_checkpoint(corrupted,root)
  except AssertionError:pass
  else:raise AssertionError('Wrong SHA accepted')
  alias=root/'alias.pth';alias.symlink_to(cp)
  try:verify_checkpoint(dict(ck,path=str(alias)),root)
  except AssertionError:pass
  else:raise AssertionError('Linked source accepted')
  child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
  try:
   expected=identity(child.pid);wrong=dict(expected,start_ticks=str(int(expected['start_ticks'])+1))
   try:open_bound_pidfd(wrong)
   except AssertionError:pass
   else:raise AssertionError('Wrong PID starttime accepted')
   fd=open_bound_pidfd(expected)
   try:
    existing=root/'existing_intent.json';existing.write_text('{}')
    try:send_once(fd,expected,existing,dict(fixture_only=True))
    except FileExistsError:pass
    else:raise AssertionError('An existing unknown intent was overwritten')
    assert child.poll() is None
    intent=root/'signal_intent.json';send_once(fd,expected,intent,dict(fixture_only=True));code=child.wait(timeout=10)
    assert code in [-2,130,1]
    try:send_once(fd,expected,intent,dict(fixture_only=True))
    except (AssertionError,FileExistsError,FileNotFoundError):pass
    else:raise AssertionError('Duplicate signal accepted')
   finally:os.close(fd)
  finally:
   if child.poll() is None:child.kill();child.wait()
 result=dict(status='LDM100K_CONTROL_NEW_CPU_FIXTURES_PASSED',metadata_negative_cases=10,wrong_pid_identity_rejected=True,
  duplicate_signal_rejected=True,actual_owned_CPU_child_signal='SIGINT',actual_owned_CPU_child_exit=code,
  production_process_signals=0,production_checkpoint_reads=0,new_gpu_contexts=0)
 (out/'SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
