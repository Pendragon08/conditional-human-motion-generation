"""Lossless verified cold storage of two retired, non-active SM-style snapshots."""
import contextlib,fcntl,gzip,hashlib,json,os,pathlib,shutil,signal,time
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'stage3_supermimic29_20261103';OLD=D/'runs/pick500_8912_72000_parallel_v2'
A=pathlib.Path('/home/ouyanghao/dreamcontrol_v2_cold_archive_20261104/retired_sm_style_v2_v3')
INDEX=R/'checkpoint_archival_20261104/retired_sm_v2_manifest.json';ROOT_FLOOR=2*1024**3+100*1024**2
EXPECTED={'resume_early_96.pt':'b346550196342fb240924e146f8e78b5eecf54a2fd6217cf0b14777d7abe04c0',
 'resume_failure_3424.pt':'8ae346f6670f3c0b7ef18c35797f0296635c7d6e9c216067ffb3507829ae2845'}
def digest(p,decompress=False):
 h=hashlib.sha256();op=gzip.open if decompress else open
 with op(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def identity(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
@contextlib.contextmanager
def exclusive_closed_source(p):
 # The kernel refuses a write lease when another process has the file open.
 # This verifies closure without needing access to foreign /proc FD tables.
 assert p.stat().st_uid==os.getuid();fd=os.open(p,os.O_RDWR);broken=[False]
 previous=signal.signal(signal.SIGIO,lambda *_:broken.__setitem__(0,True))
 try:
  fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK)
  assert fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK
  yield fd,broken
 finally:
  fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd);signal.signal(signal.SIGIO,previous)
def assert_closed(p):
 with exclusive_closed_source(p) as (fd,broken):assert not broken[0]
def atomic(p,j):
 t=p.with_suffix('.archive.tmp');t.write_text(json.dumps(j,indent=2));t.replace(p)
assert not A.exists() and not INDEX.exists()
assert json.loads((OLD/'status.json').read_text())['status']=='failed'
assert not pathlib.Path('/proc/1449777').exists(),'old optimizer unexpectedly still exists; inspect before any archive'
assert not json.loads((D/'native_universal_20261104/all_four_reference_gate.json').read_text())['training_allowed']
sources=[OLD/k for k in EXPECTED];before={str(p):identity(p) for p in sources}
assert all(not p.is_symlink() and p.resolve().is_relative_to(OLD.resolve()) and p.stat().st_uid==os.getuid() and p.stat().st_size==571807054 for p in sources)
assert shutil.disk_usage(A.parent.parent).free>=ROOT_FLOOR+sum(p.stat().st_size for p in sources)+64*1024**2
A.mkdir(parents=True,exist_ok=False);records=[];started=time.monotonic()
for p in sources:
 assert_closed(p);assert digest(p)==EXPECTED[p.name]
 target=A/(p.name+'.gz');hasher=hashlib.sha256()
 with target.open('xb') as out:
  with gzip.GzipFile(filename='',fileobj=out,mode='wb',compresslevel=3,mtime=0) as packed,p.open('rb') as src:
   for chunk in iter(lambda:src.read(4*1024**2),b''):
    assert shutil.disk_usage('/home/ouyanghao').free>=ROOT_FLOOR
    packed.write(chunk);hasher.update(chunk)
  out.flush();os.fsync(out.fileno())
 assert hasher.hexdigest()==EXPECTED[p.name] and digest(target,True)==EXPECTED[p.name]
 assert digest(p)==EXPECTED[p.name] and identity(p)==before[str(p)];assert_closed(p)
 records.append({'source':str(p),'source_sha256':EXPECTED[p.name],'source_identity':before[str(p)],'source_bytes':p.stat().st_size,
  'archive':str(target),'archive_sha256':digest(target),'archive_bytes':target.stat().st_size,'independent_full_decompression_sha256':EXPECTED[p.name],
  'status':'complete_archive_verified_source_retained'})
manifest={'status':'verified_archives_before_source_release','records':records,'source_code_sha256':digest(__file__),
 'scope':'retired failed SM-style adapter snapshots only; PPO/current diffusion/native validation weights untouched',
 'restore':'gzip-decompress archive to a new temporary file, verify source_sha256 and source_bytes, then restore original path only if absent; retain compressed archive'}
atomic(A/'manifest.json',manifest);atomic(INDEX,manifest)
for record in records:
 p=pathlib.Path(record['source']);archive=pathlib.Path(record['archive'])
 assert digest(archive)==record['archive_sha256'] and digest(archive,True)==record['source_sha256']
 assert identity(p)==record['source_identity'] and digest(p)==record['source_sha256']
 assert shutil.disk_usage('/home/ouyanghao').free>=ROOT_FLOOR
 with exclusive_closed_source(p) as (fd,broken):
  h=hashlib.sha256()
  for chunk in iter(lambda:os.read(fd,4*1024**2),b''):h.update(chunk)
  assert h.hexdigest()==record['source_sha256'] and identity(p)==record['source_identity']
  assert not broken[0] and fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK
  p.unlink()
 record['status']='source_released_complete_verified_archive_retained'
 record['no_other_open_FD_proof']='kernel F_WRLCK file lease held through final full source SHA and source release; no lease-break request observed'
 atomic(pathlib.Path(str(p)+'.archived.json'),{**record,'manifest':str(INDEX)})
 atomic(A/'manifest.json',manifest);atomic(INDEX,manifest)
manifest.update(status='retired_sm_snapshots_losslessly_cold_archived',freed_data_bytes=sum(r['source_bytes'] for r in records),
 root_free_bytes=shutil.disk_usage('/home/ouyanghao').free,data_free_bytes=shutil.disk_usage('/data').free,elapsed_seconds=time.monotonic()-started)
atomic(A/'manifest.json',manifest);atomic(INDEX,manifest);print(json.dumps(manifest),flush=True)
