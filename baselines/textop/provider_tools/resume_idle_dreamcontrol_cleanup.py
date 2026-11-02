"""Resume only the verified pre-deletion timeout; no archive/upload repetition."""
import hashlib,json,pathlib
from finish_idle_dreamcontrol import BASE,DEST,ROOT,WORK,REPOSITORY,TAG,gh,read_server,publish,reports,state
from session import remote


def main():
    prior_path=DEST/'continuation.json'
    old=json.loads(prior_path.read_text(encoding='utf-8'))
    assert old['status']=='FAILED_NO_FURTHER_DELETION' and 'TimeoutExpired' in old['error']
    saved=DEST/'continuation_timeout_attempt1.json'
    assert not saved.exists(),'Recovery has already been attempted; inspect its state'
    saved.write_bytes(prior_path.read_bytes())
    verified=read_server(ROOT+'/receipts/idle_dreamcontrol_20261026/upload_verified.json')
    proof=read_server(WORK+'/archive_index.json')
    assert verified['status']=='ALL_GITHUB_SOFTWARE_ASSET_DIGESTS_VERIFIED'
    release=json.loads(gh('api','repos/'+REPOSITORY+'/releases/tags/'+TAG))
    assert release['id']==verified['release_id']
    for item in verified['items']:
        asset=next(x for x in release['assets'] if x['name']==item['name'])
        assert asset['state']=='uploaded' and asset['size']==item['bytes'] and asset.get('digest')=='sha256:'+item['sha256']
    original_auth=read_server(WORK+'/github_verified.json')
    gh('api','repos/'+REPOSITORY+'/git/commits/'+original_auth['code_inventory_commit'])
    # Establish the previous timeout occurred before any destructive action.
    preflight=json.loads(remote(r'''
from pathlib import Path
import hashlib,json,os
root=Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
work=Path('/dev/shm/ouyanghao_dreamcontrol_backup_20261026');out=root/'receipts/idle_dreamcontrol_20261026'
assert not (out/'apply_started.json').exists() and not (out/'cleanup.json').exists()
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid==1013:
   command=(p/'cmdline').read_bytes()
   assert b'archive_idle_dreamcontrol.py' not in command,'Previous archive/apply worker still active'
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
proof=json.loads((work/'archive_index.json').read_text())
assert hashlib.sha256((root/'scripts/archive_idle_dreamcontrol.py').read_bytes()).hexdigest()==proof['source_script_sha256']
rows=json.loads((work/'source_manifest.json').read_text())
for row in rows:
 p=Path(row['path']);s=p.lstat()
 assert (s.st_ino,s.st_size,s.st_mtime_ns,s.st_uid)==(row['inode'],row['bytes'],row['mtime_ns'],1013)
print(json.dumps({'status':'PREDELETION_TIMEOUT_SOURCE_INTACT','members':len(rows),'missing_members':0,'metadata_changes':0,'destructive_phase_started':False}))
'''))
    (DEST/'resume_preflight.json').write_text(json.dumps(preflight,indent=2),encoding='utf-8')
    (DEST/'apply_timeout_attempt1.json').write_text(json.dumps({'timed_out':True,'deadline_seconds':240,'actual_child_exit_code_unavailable':True,'source_intact_after_timeout':True,'destructive_phase_started':False},indent=2),encoding='utf-8')
    commit=publish()
    state('RESUMING_VERIFIED_CLEANUP_WITH_LONGER_SUPERVISOR',resume_commit=commit,previous_failure=old['error'],error=None,error_type=None)
    # Same immutable cleanup kernel and original GitHub authorization. Only the
    # supervision deadline increases; source hashes/idleness are all rechecked.
    cleanup=json.loads(remote(r'''
from pathlib import Path
import json,subprocess
root=Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');out=root/'receipts/idle_dreamcontrol_20261026'
assert not (out/'apply_resume.log').exists()
with (out/'apply_resume.log').open('xb') as log:
 try:
  child=subprocess.run(['python3',str(root/'scripts/archive_idle_dreamcontrol.py'),'apply'],stdout=log,stderr=subprocess.STDOUT,timeout=1800)
  receipt={'actual_exit_code':child.returncode,'timed_out':False,'deadline_seconds':1800}
 except subprocess.TimeoutExpired:
  receipt={'actual_exit_code':None,'timed_out':True,'deadline_seconds':1800}
  (out/'apply_resume_exit.json').write_text(json.dumps(receipt));raise
(out/'apply_resume_exit.json').write_text(json.dumps(receipt))
assert child.returncode==0,'Cleanup failed; preserve evidence and inspect phase'
print((out/'cleanup.json').read_text())
''',timeout=1900))
    assert cleanup['status']=='IDLE_DREAMCONTROL_SOFTWARE_BACKED_UP_AND_REMOVED'
    for name in ['cleanup','apply_resume_exit']:
        value=read_server(ROOT+'/receipts/idle_dreamcontrol_20261026/'+name+'.json')
        (DEST/(name+'.json')).write_text(json.dumps(value,indent=2),encoding='utf-8')
    # All persistent recovery assets have been independently reverified above.
    # Remove only these now-redundant software archive copies from RAM, retaining
    # the complete inventory, receipts and all GMR data in their separate roots.
    ram=json.loads(remote(r'''
from pathlib import Path
import hashlib,json,os,shutil
work=Path('/dev/shm/ouyanghao_dreamcontrol_backup_20261026');root=Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
proof=json.loads((work/'archive_index.json').read_text())
assert json.loads((root/'receipts/idle_dreamcontrol_20261026/cleanup.json').read_text())['status']=='IDLE_DREAMCONTROL_SOFTWARE_BACKED_UP_AND_REMOVED'
assert work.resolve()==work and work.stat().st_uid==1013 and os.getuid()==1013
items=proof['parts']+[{'name':'idle_dreamcontrol_overlay.tar.gz','bytes':proof['archive_bytes'],'sha256':proof['archive_sha256']}]
for item in items:
 p=work/item['name'];assert p.parent==work and not p.is_symlink() and p.stat().st_uid==1013 and p.stat().st_size==item['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 assert h.hexdigest()==item['sha256']
before=shutil.disk_usage('/dev/shm').free
for item in items:(work/item['name']).unlink()
result={'status':'VERIFIED_REDUNDANT_SOFTWARE_ARCHIVE_RAM_COPIES_REMOVED','bytes_removed':sum(x['bytes'] for x in items),'removed_names':[x['name'] for x in items],'ram_free_before':before,'ram_free_after':shutil.disk_usage('/dev/shm').free,'datasets_deleted':0,'checkpoints_deleted':0,'manifest_and_receipts_retained':True}
(root/'receipts/idle_dreamcontrol_20261026/ram_archive_cleanup.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
''',timeout=180))
    (DEST/'ram_archive_cleanup.json').write_text(json.dumps(ram,indent=2),encoding='utf-8')
    reports(proof,cleanup)
    state('IDLE_SOFTWARE_RECOVERY_AND_CLEANUP_VERIFIED',free_after=cleanup['free_after'],release_url=release['html_url'],datasets_deleted=0,checkpoints_deleted=0,records_deleted=0,error=None,error_type=None)
    publish()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        state('RESUME_FAILED_RETAIN_ALL_EVIDENCE',error_type=type(error).__name__,error=str(error)[-700:])
        raise
