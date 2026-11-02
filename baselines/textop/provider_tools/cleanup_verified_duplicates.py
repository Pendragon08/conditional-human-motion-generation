"""Remove two fixed redundant copies only after recoverability/GitHub checks.

Keep authoritative datasets, complete original archives, checkpoints and records.
The CLoSD target is a temporary restore-verification copy, not the server dataset.
"""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,sys
from session import REPO,GH,remote
from restore_indexed_cache import sha
BASE=REPO/'baselines/textop';DEST=BASE/'results/data_preparation_20261025'
TASK=REPO.parent.parent/'textop_reproduction_20261024'
CMU=TASK/'private_assets/incoming_amass/CMU_smplx_g.tar.bz2'
PART=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives/CMU_smplx_g.tar.bz2.part')
OLD=pathlib.Path('D:/具身智能/服务器磁盘清理_20261022/storage_cleanup_20261022_102452')
COPY=OLD/'restored_closd_generated_backup/conditional-human-motion-generation/results/closd_baseline_20261010/fresh_generation/CLoSD.pkl'
SAVED=OLD/'closd_generated_20261010_backup_index.json'
sys.stdout.reconfigure(encoding='utf-8')

def prefix_sha(p,count):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while count:
   b=f.read(min(count,8*1024**2));assert b;h.update(b);count-=len(b)
 return h.hexdigest()

def verify():
 cmu=json.loads((DEST/'cmu_resume.json').read_text(encoding='utf-8'))
 assert cmu['status']=='LOCAL_ORIGINAL_ARCHIVE_VERIFIED' and cmu['compressed_crc_and_tar_headers_verified']
 assert sha(CMU)==cmu['sha256'] and CMU.stat().st_size==cmu['archive_bytes']
 assert PART.is_file() and not PART.is_symlink() and PART.stat().st_size==5817466303
 partial_sha=sha(PART);assert prefix_sha(CMU,PART.stat().st_size)==partial_sha
 saved=json.loads(SAVED.read_text(encoding='utf-8'));original=saved['files'][0]
 assert COPY.is_file() and not COPY.is_symlink() and COPY.stat().st_size==original['bytes'] and sha(COPY)==original['sha256']
 server=json.loads(remote('''
import pathlib,hashlib,json,os
p=pathlib.Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/closd_baseline_20261010/fresh_generation/CLoSD.pkl')
assert os.getuid()==1013 and p.stat().st_uid==1013 and not p.is_symlink()
before=p.stat();h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert p.stat().st_mtime_ns==before.st_mtime_ns and p.stat().st_size==before.st_size
print(json.dumps({'bytes':before.st_size,'sha256':h.hexdigest(),'source_dataset_retained':True}))
''',timeout=180))
 assert server['sha256']==original['sha256'] and server['bytes']==original['bytes']
 env=os.environ.copy();env.update(HTTPS_PROXY='http://127.0.0.1:7897',HTTP_PROXY='http://127.0.0.1:7897')
 release=json.loads(subprocess.check_output([GH,'api','repos/Pendragon08/conditional-human-motion-generation/releases/tags/server-storage-archive-20261022'],env=env))
 lookup={x['name']:x for x in release['assets']};pieces=[]
 for x in saved['pieces_in_order']:
  local=OLD/x['name'];assert local.is_file() and local.stat().st_size==x['bytes'] and sha(local)==x['sha256']
  asset=lookup[x['name']];assert asset['state']=='uploaded' and asset['size']==x['bytes']
  assert asset.get('digest')=='sha256:'+x['sha256'],'GitHub archive digest unavailable or different'
  pieces.append({'file':x['name'],'bytes':x['bytes'],'sha256':x['sha256'],'github_asset_id':asset['id']})
 result={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
  'status':'TWO_DUPLICATE_COPIES_RECOVERY_VERIFIED',
  'targets':[{'path':str(PART.resolve()),'bytes':PART.stat().st_size,'sha256':partial_sha,'retained_source':'complete verified C-drive CMU archive','restore':'copy first5817466303 bytes from retained complete archive'},
             {'path':str(COPY.resolve()),'bytes':COPY.stat().st_size,'sha256':original['sha256'],'retained_source':'verified original server CLoSD dataset plus local and GitHub archive pieces','restore':'retained original server path or ordered archive pieces'}],
  'retained_complete_CMU_sha256':cmu['sha256'],'retained_CLoSD_server':server,'retained_CLoSD_archive_pieces':pieces,
  'authoritative_datasets_removed':0,'checkpoints_removed':0,'unique_records_removed':0}
 return result

def main(commit=None):
 proof=verify()
 if not commit:
  (DEST/'duplicate_copy_recovery_verified.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
  print(json.dumps({k:v for k,v in proof.items() if k not in ['targets','retained_CLoSD_archive_pieces']}),flush=True);return
 subprocess.run([GH,'api','repos/Pendragon08/conditional-human-motion-generation/git/commits/'+commit],check=True,stdout=subprocess.DEVNULL)
 for rel in ['provider_tools/cleanup_verified_duplicates.py','results/data_preparation_20261025/duplicate_copy_recovery_verified.json']:
  old=subprocess.check_output(['git','-C',str(REPO),'show',commit+':baselines/textop/'+rel]);current=(BASE/rel).read_bytes()
  assert json.loads(old)==json.loads(current) if rel.endswith('.json') else old==current
 before=shutil.disk_usage('D:/').free
 for p,row in zip([PART,COPY],proof['targets']):
  exact=p.resolve();assert str(exact)==row['path'] and pathlib.Path('D:/具身智能').resolve() in exact.parents
  assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256'];p.unlink()
 result={**proof,'status':'REDUNDANT_PARTIAL_AND_RESTORE_TEST_COPY_REMOVED','D_free_before':before,'D_free_after':shutil.disk_usage('D:/').free,
         'backed_up_recovery_code_and_evidence_commit':commit,'redundant_partial_copies_removed':1,'duplicate_dataset_verification_copies_removed':1}
 (DEST/'duplicate_copy_cleanup.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 (pathlib.Path('D:/具身智能/TextOp复现_20261024')/'重复副本清理核验_20261025.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ['targets','retained_CLoSD_archive_pieces']}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--apply-backed-up-commit');a=p.parse_args();main(a.apply_backed_up_commit)
