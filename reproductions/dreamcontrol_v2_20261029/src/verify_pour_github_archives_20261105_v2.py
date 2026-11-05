"""Two complete readbacks of archived completed Pour intermediates; never deletes sources."""
import fcntl,hashlib,json,os,re,stat,sys,time
from pathlib import Path
from github_archive_stream_20261105_v2 import get_asset,assert_asset,inspect_archive

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
P=R/'publication_snapshot_20261105';J=R/'stage3_supported_references_20261104/task_ppo_20261104'
OUT=R/'checkpoint_archival_20261104/github_pour_readback_20261105_v1'
FINAL=J/'pour/ppo_8912_5000_v1/ppo/model_4999.pt'
FINAL_SHA='442f0453f6a4f8d66e0dee9d15f1466f722c4366aba739bb0f41e4e1416f87ff'
ASSETS=[{'id':712860202,'name':'dreamcontrol-v2-remote-models-20261105-part02.tar','bytes':520439808,'sha256':'1c65264f98563e5bbaa9dbd91174e650e6800cec6bc7fe4cc542f42072e76cf2'}, {'id':712850886,'name':'dreamcontrol-v2-remote-models-20261105-part01.tar','bytes':1057892352,'sha256':'82f5d013ad950de329591dc29d1f7ad517596447a0859e95317cb2bd72fc8cee'}]
PREFIX='stage3_supported_references_20261104/task_ppo_20261104/pour/ppo_8912_5000_v1/ppo/'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def identity(s):
 if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or s.st_nlink!=1:raise RuntimeError('Regular owned single-link source required')
 return {'bytes':s.st_size,'mtime_ns':str(s.st_mtime_ns),'inode':str(s.st_ino),'device':str(s.st_dev),'uid':s.st_uid,'nlink':s.st_nlink}
def save(p,obj):
 temp=p.with_name(p.name+'.tmp')
 with temp.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def fixed_manifest(a):return (P/'remote_upload_v1'/(a['name']+'.manifest.json')).read_bytes()
def check_cancelled():
 cancel=J/'user_cancelled_after_pour_20261105_v1/receipt.json';c=json.loads(cancel.read_text())
 if c['status']!='cancelled_by_user_after_pour_completion' or c['remaining_same_identity_processes']:raise RuntimeError('User cancellation receipt does not confirm queue stop')
 for proc in c['stopped_processes']:
  s=Path('/proc')/str(proc['pid'])/'stat'
  if s.exists():
   after=s.read_text().rsplit(')',1)[1].split()
   if after[19]==proc['start_ticks']:raise RuntimeError('Cancelled process identity still exists')
 training=json.loads((J/'pour/ppo_8912_5000_v1/status.json').read_text())
 if training['status']!='ppo_completed' or training['policy_updates']!=5000 or not training['checkpoint_roundtrip_verified']:raise RuntimeError('Pour not completed')
 if digest(FINAL)!=FINAL_SHA:raise RuntimeError('Protected Pour final SHA mismatch')
 return cancel,c
def build_candidates():
 cancel,_=check_cancelled();rows=[]
 for a in ASSETS:
  for r in json.loads(fixed_manifest(a))['files']:
   match=re.fullmatch(re.escape(PREFIX)+r'model_(\d+)\.pt',r['path'])
   if not match:continue
   number=int(match[1])
   if number%50!=0 or not 0<=number<=4450:raise RuntimeError('Unexpected candidate range')
   p=R/r['path'];before=identity(os.stat(p,follow_symlinks=False));fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
   try:
    if identity(os.fstat(fd))!=before:raise RuntimeError('Opened identity mismatch')
    h=hashlib.sha256()
    for b in iter(lambda:os.read(fd,4*1024**2),b''):h.update(b)
    if identity(os.fstat(fd))!=before:raise RuntimeError('Source changed while hashing')
   finally:os.close(fd)
   if identity(os.stat(p,follow_symlinks=False))!=before or h.hexdigest()!=r['sha256'] or before['bytes']!=r['bytes']:raise RuntimeError('Source differs from archive manifest')
   rows.append({'source_path':str(p),'member':r['path'],'bytes':r['bytes'],'sha256':r['sha256'],'identity':before,'asset_id':a['id'],'asset_name':a['name']})
 rows.sort(key=lambda r:int(re.search(r'(\d+)\.pt$',r['member'])[1]))
 if len(rows)!=90 or len({r['source_path'] for r in rows})!=90:raise RuntimeError('Expected exactly 90 distinct candidates')
 return {'schema_version':1,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'Only completed non-final Pour model_0.pt through model_4450.pt at increments of50. No source released.','rows':rows,'source_bytes':sum(r['bytes'] for r in rows),'assets':ASSETS,'fixed_manifest_sha256':{str(a['id']):hashlib.sha256(fixed_manifest(a)).hexdigest() for a in ASSETS},'cancel_receipt':str(cancel),'cancel_receipt_sha256':digest(cancel),'protected_final':str(FINAL),'protected_final_sha256':FINAL_SHA}
def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'verification.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 token=json.loads(sys.stdin.buffer.read())['token'];plan_path=OUT/'candidates_manifest.json'
 if not plan_path.exists():save(plan_path,build_candidates())
 plan=json.loads(plan_path.read_text());plan_sha=digest(plan_path)
 if plan['assets']!=ASSETS or len(plan['rows'])!=90 or plan['source_bytes']!=550945092:raise RuntimeError('Candidate manifest scope changed')
 state={'status':'readback_in_progress_no_sources_released','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'candidate_manifest':str(plan_path),'candidate_manifest_sha256':plan_sha,'total_candidates':90,'candidate_source_bytes':plan['source_bytes'],'passes':[],'verified_candidate_count':0,'sources_released':False,'verifier_sha256':digest(__file__),'stream_verifier_sha256':digest(Path(__file__).with_name('github_archive_stream_20261105_v2.py'))}
 prior=OUT/'verification_receipt.json'
 if prior.exists():
  old=prior.read_bytes();history=OUT/('prior_attempt_'+str(time.time_ns())+'.json');history.write_bytes(old);state['prior_attempt_receipt']=str(history);state['prior_attempt_receipt_sha256']=hashlib.sha256(old).hexdigest()
 save(OUT/'verification_receipt.json',state)
 try:
  for asset in ASSETS:
   mb=fixed_manifest(asset)
   if hashlib.sha256(mb).hexdigest()!=plan['fixed_manifest_sha256'][str(asset['id'])]:raise RuntimeError('Fixed archive manifest changed')
   expected={r['path']:(r['bytes'],r['sha256']) for r in json.loads(mb)['files']}
   for pass_no in [1,2]:
    target=OUT/f"asset_{asset['id']}_pass{pass_no}.json"
    state.update(current_asset_id=asset['id'],current_pass=pass_no,read_bytes=0);save(OUT/'verification_receipt.json',state)
    def progress(n):state.update(read_bytes=n,progress_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save(OUT/'verification_receipt.json',state)
    if target.exists():
     result=json.loads(target.read_text());actual={k:(v['bytes'],v['sha256']) for k,v in result['members'].items()}
     if actual!=expected or result['archive_sha256']!=asset['sha256'] or result['archive_bytes']!=asset['bytes'] or result['candidate_manifest_sha256']!=plan_sha:raise RuntimeError('Existing pass receipt mismatch; preserve and inspect')
     assert_asset(get_asset(token,asset['id']),asset)
    else:
     result=inspect_archive(token,asset,mb,progress);result.update(pass_number=pass_no,candidate_manifest_sha256=plan_sha,source_released=False);save(target,result)
    state['passes'].append({'asset_id':asset['id'],'pass_number':pass_no,'receipt':str(target),'receipt_sha256':digest(target),'archive_bytes':result['archive_bytes'],'archive_sha256':result['archive_sha256'],'member_count':result['member_count']});save(OUT/'verification_receipt.json',state)
    print(json.dumps({'asset_id':asset['id'],'pass':pass_no,'verified_bytes':asset['bytes'],'members':result['member_count']}),flush=True)
   state['verified_candidate_count']+=sum(r['asset_id']==asset['id'] for r in plan['rows']);save(OUT/'verification_receipt.json',state)
  check_cancelled();state.update(status='two_complete_readbacks_all90_pour_intermediates_verified_no_sources_released',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protected_final_rechecked_sha256=digest(FINAL));save(OUT/'verification_receipt.json',state)
  print(json.dumps({'status':state['status'],'candidates':90,'preserved_bytes':plan['source_bytes']}),flush=True)
 except BaseException as e:
  state.update(status='readback_failed_preserved_no_source_release',error_type=type(e).__name__,error=str(e).replace(token,'[REDACTED]')[:800]);save(OUT/'verification_receipt.json',state);raise
 finally:token='';lock.close()
if __name__=='__main__':main()
