"""Explicit, reviewable release of strictly enumerated archived Pour intermediates.

Default is check-only. --execute --count N is required for source release.
Never modifies final weights, archived bytes, task data, or training controllers.
"""
import argparse,fcntl,hashlib,json,os,signal,stat,sys,time
from pathlib import Path
from github_archive_stream_20261105_v2 import get_asset,assert_asset
from verify_pour_github_archives_20261105_v2 import R,P,J,OUT,FINAL,FINAL_SHA,ASSETS,identity,digest,save,check_cancelled,fixed_manifest

def journal(path,row):
 with path.open('a') as f:json.dump(row,f,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
def fsync_dir(path):
 fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def validate_evidence(token):
 receipt_path=OUT/'verification_receipt.json';receipt=json.loads(receipt_path.read_text());plan_path=OUT/'candidates_manifest.json';plan=json.loads(plan_path.read_text())
 if receipt['status']!='two_complete_readbacks_all90_pour_intermediates_verified_no_sources_released' or receipt['sources_released'] or receipt['verified_candidate_count']!=90:raise RuntimeError('Two complete verified readbacks required')
 if digest(plan_path)!=receipt['candidate_manifest_sha256'] or plan['assets']!=ASSETS or len(plan['rows'])!=90 or plan['source_bytes']!=550945092:raise RuntimeError('Fixed candidate manifest mismatch')
 if digest(Path(__file__).with_name('verify_pour_github_archives_20261105_v2.py'))!=receipt['verifier_sha256'] or digest(Path(__file__).with_name('github_archive_stream_20261105_v2.py'))!=receipt['stream_verifier_sha256']:raise RuntimeError('Verification code identity mismatch')
 proofs={(p['asset_id'],p['pass_number']):p for p in receipt['passes']}
 if len(proofs)!=4:raise RuntimeError('Exactly four independent pass receipts required')
 for a in ASSETS:
  assert_asset(get_asset(token,a['id']),a)
  mb=fixed_manifest(a);manifest=json.loads(mb);expected={r['path']:(r['bytes'],r['sha256']) for r in manifest['files']}
  if hashlib.sha256(mb).hexdigest()!=plan['fixed_manifest_sha256'][str(a['id'])]:raise RuntimeError('Fixed embedded manifest hash mismatch')
  for number in [1,2]:
   p=proofs[(a['id'],number)];path=Path(p['receipt'])
   if path!=OUT/f"asset_{a['id']}_pass{number}.json" or digest(path)!=p['receipt_sha256']:raise RuntimeError('Pass receipt integrity mismatch')
   q=json.loads(path.read_text());actual={k:(v['bytes'],v['sha256']) for k,v in q['members'].items()}
   if q['pass_number']!=number or q['candidate_manifest_sha256']!=receipt['candidate_manifest_sha256'] or q['archive_bytes']!=a['bytes'] or q['archive_sha256']!=a['sha256'] or actual!=expected or not q['all_member_sha256_verified'] or not q['whole_tar_sha256_verified'] or not q['metadata_verified_before_and_after']:raise RuntimeError('Incomplete/incorrect archive readback proof')
 for i,row in enumerate(plan['rows']):
  expected_path=J/f'pour/ppo_8912_5000_v1/ppo/model_{i*50}.pt'
  if Path(row['source_path'])!=expected_path or row['member']!=str(expected_path.relative_to(R)) or row['source_path']==str(FINAL):raise RuntimeError('Candidate path outside exact whitelist')
  asset=next(a for a in ASSETS if a['id']==row['asset_id']);stored={r['path']:r for r in json.loads(fixed_manifest(asset))['files']}.get(row['member'])
  if stored is None or (row['bytes'],row['sha256'])!=(stored['bytes'],stored['sha256']):raise RuntimeError('Candidate not in verified archive')
 cancel,_=check_cancelled()
 if str(cancel)!=plan['cancel_receipt'] or digest(cancel)!=plan['cancel_receipt_sha256']:raise RuntimeError('Cancellation evidence changed')
 return plan,receipt_path,receipt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--count',type=int);args=ap.parse_args()
 if args.execute and args.count is None:raise RuntimeError('Explicit reviewed --count required for release')
 count=args.count or 90
 if not 1<=count<=90:raise RuntimeError('Count must be 1..90')
 token=json.loads(sys.stdin.buffer.read())['token'];verify_lock=(OUT/'verification.lock').open('a');fcntl.flock(verify_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 lock=(OUT/'release.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 plan,evidence_path,evidence=validate_evidence(token);token='';selected=plan['rows'][:count]
 before=os.statvfs('/data');before_free=before.f_bavail*before.f_frsize
 receipt={'status':'release_in_progress' if args.execute else 'check_only_in_progress','source_release_requested':args.execute,'count':count,'candidate_manifest_sha256':evidence['candidate_manifest_sha256'],'verification_receipt_sha256':digest(evidence_path),'release_script_sha256':digest(__file__),'free_before':before_free,'released':[],'checked':[],'protected_final_sha256_before':digest(FINAL)}
 target=OUT/('source_release_receipt.json' if args.execute else 'source_release_check_receipt.json');save(target,receipt)
 released_log=OUT/'source_releases.jsonl';broken=[False];signal.signal(signal.SIGIO,lambda *_:broken.__setitem__(0,True))
 try:
  for row in selected:
   p=Path(row['source_path']);marker=Path(str(p)+'.archived.json');pending=Path(str(p)+'.archived.json.pending-github-v1')
   if p.is_symlink() or p.parent.resolve()!=J/'pour/ppo_8912_5000_v1/ppo':raise RuntimeError('Symlink/noncanonical candidate path')
   asset=next(a for a in ASSETS if a['id']==row['asset_id'])
   marker_value={'status':'complete_file_bytes_preserved_in_verified_github_archive','source_path':str(p),'source_sha256':row['sha256'],'source_bytes':row['bytes'],'source_identity':row['identity'],'repository':'Pendragon08/conditional-human-motion-generation','release_id':434987063,'asset_id':asset['id'],'archive_name':asset['name'],'archive_bytes':asset['bytes'],'archive_sha256':asset['sha256'],'archive_member':row['member'],'verified_two_full_readbacks':True,'candidate_manifest':str(OUT/'candidates_manifest.json'),'candidate_manifest_sha256':evidence['candidate_manifest_sha256'],'verification_receipt':str(evidence_path),'verification_receipt_sha256':digest(evidence_path),'restoration_script':str(P/'restore_pour_github_archive_20261105_v2.py'),'kernel_F_WRLCK_lease_held_through_source_release':True}
   if not p.exists():
    if not marker.exists():raise RuntimeError('Missing source without completed archive marker; preserve and inspect journal')
    if json.loads(marker.read_text())!=marker_value:raise RuntimeError('Existing archive marker differs')
    receipt['released'].append({'path':str(p),'bytes':row['bytes'],'sha256':row['sha256'],'already_released':True});save(target,receipt);continue
   if marker.exists() or pending.exists():raise RuntimeError('Source/marker conflict; inspect without overwriting')
   if identity(os.stat(p,follow_symlinks=False))!=row['identity']:raise RuntimeError('Final source identity changed')
   fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW);leased=False;broken[0]=False
   try:
    if identity(os.fstat(fd))!=row['identity']:raise RuntimeError('Opened source identity mismatch')
    fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK);leased=True
    if fcntl.fcntl(fd,fcntl.F_GETLEASE)!=fcntl.F_WRLCK:raise RuntimeError('Exclusive kernel lease unavailable')
    h=hashlib.sha256()
    for b in iter(lambda:os.read(fd,4*1024**2),b''):
     if broken[0]:raise RuntimeError('Kernel lease break requested; stop before source release')
     h.update(b)
    if h.hexdigest()!=row['sha256'] or identity(os.fstat(fd))!=row['identity'] or identity(os.stat(p,follow_symlinks=False))!=row['identity'] or broken[0]:raise RuntimeError('Final full SHA/source identity/lease check failed')
    receipt['checked'].append({'path':str(p),'bytes':row['bytes'],'sha256':row['sha256'],'exclusive_kernel_lease_acquired':True})
    if args.execute:
     with pending.open('x') as f:json.dump(marker_value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
     journal(released_log,{'event':'prepared_verified_source_release','source':row,'pending_marker':str(pending),'verification_receipt_sha256':digest(evidence_path)})
     if broken[0] or fcntl.fcntl(fd,fcntl.F_GETLEASE)!=fcntl.F_WRLCK or identity(os.fstat(fd))!=row['identity'] or identity(os.stat(p,follow_symlinks=False))!=row['identity']:raise RuntimeError('Lease or identity changed at unlink boundary')
     os.unlink(p);os.replace(pending,marker);fsync_dir(p.parent)
     journal(released_log,{'event':'released_under_exclusive_kernel_F_WRLCK_lease','source_path':str(p),'bytes':row['bytes'],'sha256':row['sha256'],'marker':str(marker)})
     receipt['released'].append({'path':str(p),'bytes':row['bytes'],'sha256':row['sha256'],'already_released':False})
   finally:
    if leased:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK)
    os.close(fd)
   save(target,receipt)
  if digest(FINAL)!=FINAL_SHA:raise RuntimeError('Protected final changed')
  free=os.statvfs('/data');receipt.update(status='verified_archived_intermediates_released' if args.execute else 'check_only_passed_no_sources_released',released_source_bytes=sum(r['bytes'] for r in receipt['released']),free_after=free.f_bavail*free.f_frsize,protected_final_sha256_after=FINAL_SHA,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save(target,receipt)
  print(json.dumps({'status':receipt['status'],'count':count,'source_bytes':sum(r['bytes'] for r in selected),'released_bytes':receipt['released_source_bytes'],'receipt':str(target)}),flush=True)
 except BaseException as e:
  receipt.update(status='release_or_check_failed_preserved',error_type=type(e).__name__,error=str(e)[:800]);save(target,receipt);raise
 finally:lock.close();verify_lock.close()
if __name__=='__main__':main()
