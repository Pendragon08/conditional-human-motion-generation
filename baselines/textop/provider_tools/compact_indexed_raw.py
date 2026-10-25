"""Transparent NTFS compression of the exact private indexed raw working set.

Only physical storage changes. Original paths, logical bytes and SHA256 values
must remain identical. No deletion, archive extraction or OS-wide setting change.
"""
import argparse,ctypes,datetime,hashlib,json,pathlib,shutil,subprocess,sys
from session import REPO
sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
BASE=REPO/'baselines/textop';TASK=REPO.parent.parent/'textop_reproduction_20261024'
ROOT=(TASK/'private_assets/indexed_raw_expansion_v1').resolve()
GET=ctypes.WinDLL('kernel32',use_last_error=True).GetCompressedFileSizeW
GET.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)];GET.restype=ctypes.c_ulong

def physical(path):
    hi=ctypes.c_ulong(0);ctypes.set_last_error(0);lo=GET(str(path),ctypes.byref(hi))
    if lo==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
    return (hi.value<<32)|lo

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main(args):
    index=ROOT/'ready.private.json';rows=json.loads(index.read_text(encoding='utf-8'))
    assert len(rows)==1653 and ROOT.parent==(TASK/'private_assets').resolve()
    if args.sample:rows=rows[:1]
    paths=[pathlib.Path(x['local_raw_path']).resolve() for x in rows]
    assert all(p.parent==ROOT and p.is_file() and not p.is_symlink() for p in paths)
    before=sum(physical(p) for p in paths);logical=sum(p.stat().st_size for p in paths);free=shutil.disk_usage(ROOT).free
    codes=[]
    log=TASK/('ntfs_compression_sample.log' if args.sample else 'ntfs_compression_full.log')
    with log.open('ab') as output:
        for start in range(0,len(paths),40):
            p=subprocess.run(['compact.exe','/C','/Q','/I',*[str(x) for x in paths[start:start+40]]],stdout=output,stderr=subprocess.STDOUT)
            codes.append(p.returncode)
    for row,path in zip(rows,paths):assert sha(path)==row['raw_sha256'],'Logical content checksum changed: '+path.name
    assert sum(p.stat().st_size for p in paths)==logical
    after=sum(physical(p) for p in paths)
    receipt={'scope':'exact indexed raw1653 working set only; one file for sample mode','sample':args.sample,'files_verified':len(paths),'logical_bytes':logical,'physical_bytes_before':before,'physical_bytes_after':after,'physical_bytes_saved':before-after,'C_free_before':free,'C_free_after':shutil.disk_usage(ROOT).free,'returncodes':sorted(set(codes)),'all_sha256_unchanged':True,'files_deleted':0,'source_archives_modified':False,'os_wide_settings_modified':False,'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}
    name='ntfs_compression_sample.json' if args.sample else 'ntfs_compression_full.json'
    (BASE/'results/data_preparation_20261025'/name).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (TASK/name).write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sample',action='store_true');main(p.parse_args())
