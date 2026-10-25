"""Lossless NTFS compression of an explicit list of retained personal records.

No records, datasets or checkpoints are deleted. SHA256 and logical lengths
must remain unchanged; only already existing files in two named roots qualify.
"""
import datetime, hashlib, json, pathlib, shutil, subprocess, sys
from session import REPO
from compact_indexed_raw import physical

sys.stdout.reconfigure(encoding='utf-8')
TASK=REPO.parent.parent/'textop_reproduction_20261024'
BASE=REPO/'baselines/textop'
ROOTS=[pathlib.Path('D:/具身智能/服务器磁盘清理_20261022').resolve(),
       pathlib.Path('D:/具身智能/conditional-human-motion-generation/results').resolve()]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    rows=json.loads((TASK/'record_compression_allowlist.private.json').read_text(encoding='utf-8'))
    assert len(rows)==13
    before=shutil.disk_usage('D:/').free;details=[]
    for row in rows:
        p=pathlib.Path(row['path']).resolve()
        assert any(p.is_relative_to(r) for r in ROOTS)
        assert p.is_file() and not p.is_symlink() and p.suffix in ['.json','.pkl']
        stat=p.stat();assert stat.st_size==row['bytes']
        h=sha(p);allocated=physical(p)
        with (TASK/'retained_record_compression.log').open('ab') as output:
            proc=subprocess.run(['compact.exe','/C','/Q',str(p)],stdout=output,stderr=subprocess.STDOUT)
        assert proc.returncode==0
        assert p.stat().st_size==stat.st_size and p.stat().st_mtime_ns==stat.st_mtime_ns
        assert sha(p)==h
        details.append({'file':p.name,'sha256':h,'logical_bytes':stat.st_size,
                        'physical_before':allocated,'physical_after':physical(p)})
        print(json.dumps({'verified_files':len(details),'saved_bytes_so_far':sum(x['physical_before']-x['physical_after'] for x in details)}),flush=True)
    result={'time_beijing':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'scope':'13 explicitly listed retained files in personal D drive roots',
            'files_deleted':0,'all_sha256_unchanged':True,'details':details,
            'physical_bytes_saved':sum(x['physical_before']-x['physical_after'] for x in details),
            'D_free_before':before,'D_free_after':shutil.disk_usage('D:/').free}
    (BASE/'results/data_preparation_20261025/retained_record_compression.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (pathlib.Path('D:/具身智能/TextOp复现_20261024')/'保留记录无损压缩_20261025.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
