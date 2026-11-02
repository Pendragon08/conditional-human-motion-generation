"""Standard-library identity and immutable-checkpoint checks for user early stop."""
import hashlib,json,os,pathlib,select,signal,stat

RESTORE_KEYS=['model_restored_exactly','optimizer_restored_exactly','ema_restored_exactly',
 'rng_and_generator_restored_exactly','clock_restored_exactly','pending_evaluation_fields_restored']
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(pathlib.Path(path).read_bytes())
def write_new(path,data):
 p=pathlib.Path(path)
 with p.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def identity(pid):
 p=pathlib.Path('/proc')/str(pid);fields=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,start_ticks=fields[19],uid=p.stat().st_uid,
  argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x],state=fields[0],pgid=int(fields[2]))
def verify_identity(expected):
 actual=identity(expected['pid'])
 assert all(actual[k]==expected[k] for k in ['pid','start_ticks','uid','argv']),('Process identity differs',actual)
 assert actual['state'] not in ['Z','X']
 return actual
def open_bound_pidfd(expected):
 verify_identity(expected);fd=os.pidfd_open(expected['pid'],0)
 try:
  verify_identity(expected);poll=select.poll();poll.register(fd,select.POLLIN)
  assert not poll.poll(0),'Bound process has already exited'
 except BaseException:os.close(fd);raise
 return fd
def checkpoint_ready(receipt,policy):
 assert receipt['spec_sha256']==policy['original_spec_sha256']
 assert receipt['source_sha256']==policy['original_entry_sha256']
 assert receipt['runtime_binding_sha256']==policy['original_runtime_binding_sha256']
 for k in ['corpus_receipt_sha256','data_acceptance_sha256','strict_audit_receipt_sha256']:assert receipt[k]==policy[k]
 assert receipt['mode']=='train'
 assert receipt['completed_primitive_attempts']==receipt['optimizer_updates']+receipt['skipped_updates']
 target=policy['target_primitive_attempts'];assert target==100000
 candidates=[x for x in receipt.get('checkpoints',[]) if x['manager_step']==target]
 if not candidates:return None
 assert len(candidates)==1
 ck=candidates[0]
 assert ck['path']==policy['checkpoint_path'] and ck['manager_step']==target
 assert 0<=ck['actual_optimizer_updates']<=target
 assert 0<ck['bytes']<=policy['checkpoint_upper_bound_bytes']
 assert len(ck['sha256'])==64 and all(x in '0123456789abcdef' for x in ck['sha256'])
 assert all(receipt.get('latest_file_resume_verification',{}).get(k) is True for k in RESTORE_KEYS)
 assert receipt['completed_primitive_attempts']>=target
 return ck
def verify_checkpoint(ck,allowed_root):
 p=pathlib.Path(ck['path']);root=pathlib.Path(allowed_root)
 assert p.is_absolute() and p.resolve()==p and root.resolve()==root and root in p.parents
 here=p
 while True:
  assert here.lstat().st_uid==os.getuid() and not here.is_symlink()
  if here==root:break
  here=here.parent
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size==ck['bytes']
 fingerprint=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns,x.st_uid)
 before=fingerprint(s);assert sha(p)==ck['sha256'] and fingerprint(p.lstat())==before
 return dict(path=str(p),bytes=s.st_size,sha256=ck['sha256'],device=s.st_dev,inode=s.st_ino,
  mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,owner_uid=s.st_uid,checkpoint_payload_deserialized=False)
def send_once(fd,expected,intent_path,context):
 verify_identity(expected)
 write_new(intent_path,dict(status='ONE_BOUND_PIDFD_SIGINT_DISPATCH_INTENT',signal=int(signal.SIGINT),process=expected,**context))
 # An intent without a sent/result record is UNKNOWN, never permission to resend.
 signal.pidfd_send_signal(fd,signal.SIGINT,None,0)
