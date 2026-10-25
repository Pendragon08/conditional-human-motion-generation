"""Read-only provenance check plus one fresh extraction witness before cleanup."""
import datetime,json,pathlib,shutil,sys
from session import REPO,remote
from restore_indexed_cache import sha,restore,ARCHIVES
from compact_indexed_raw import physical
sys.stdout.reconfigure(encoding='utf-8')
BASE=REPO/'baselines/textop';TASK=REPO.parent.parent/'textop_reproduction_20261024'
CACHE=(TASK/'private_assets/indexed_raw_expansion_v1').resolve()
ADIR=pathlib.Path('D:/具身智能/PhyGile复现/后续AMASS下载_20261021/archives')
index=CACHE/'ready.private.json'
rows=json.loads(index.read_text(encoding='utf-8'));assert len(rows)==1653
originals=[]
for digest,name in ARCHIVES.items():
    p=ADIR/name;assert p.is_file() and sha(p)==digest
    originals.append({'file':name,'bytes':p.stat().st_size,'sha256':digest})
    print(json.dumps({'original_archive_verified':name}),flush=True)
files=[]
for row in rows:
    p=pathlib.Path(row['local_raw_path']).resolve()
    assert p.parent==CACHE and p.is_file() and not p.is_symlink()
    assert sha(p)==row['raw_sha256']
    files.append({'path':str(p),'sha256':row['raw_sha256'],'bytes':p.stat().st_size,'physical_bytes':physical(p)})
proof=json.loads(remote('''
import pathlib,json,hashlib,os,time
assert os.getuid()==1013
r=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/local1653_v1')
rows=json.loads((r/'pairs.private.json').read_text());assert len(rows)==1653
for row in rows:
 p=pathlib.Path(row['raw_stageii_candidates'][0]);assert p.parent==r/'raw' and p.stat().st_uid==1013
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['raw_sha256']
p=r/'prepared1653_v1/progress.json'
progress=json.loads(p.read_text())
print(json.dumps({'files_verified':len(rows),'raw_working_copy_bytes':sum(pathlib.Path(x['raw_stageii_candidates'][0]).stat().st_size for x in rows),'progress':{k:progress.get(k) for k in ['status','completed_sequences','failed_sequences']},'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
''',timeout=180))
out=TASK/'private_assets/cache_recovery_witness_v1';assert not out.exists()
witness=restore(index,ADIR,out,limit=1)
receipt={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'purpose':'only redundant selected extraction cache may be evicted; complete original datasets retained',
 'cache_files_verified':len(files),'cache_logical_bytes':sum(x['bytes'] for x in files),
 'cache_physical_bytes':sum(x['physical_bytes'] for x in files),
 'index_sha256':sha(index),'retained_original_archives':originals,'server_working_copy':proof,
 'fresh_archive_restore_witness':witness,'files_deleted':0,
 'restore_helper_sha256':sha(pathlib.Path(__file__).with_name('restore_indexed_cache.py'))}
(TASK/'cache_eviction_plan.private.json').write_text(json.dumps({'receipt':receipt,'files':files}),encoding='utf-8')
(BASE/'results/data_preparation_20261025/cache_recovery_verified.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt),flush=True)
