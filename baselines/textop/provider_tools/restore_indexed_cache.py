"""Rebuild private selected extraction cache from retained licensed originals.

Requires the private ready index; neither motion bytes nor captions belong in
Git. Stream tar members to exact hashed cache names; no extractall/path traversal.
"""
import argparse, hashlib, json, pathlib, shutil, tarfile

ARCHIVES={
 'b8c5fca189bdb52768cdef8b93a3892a9e0788100b4553aa1ec63eac1ddf66c0':
 'BMLmovi_smplx_g.tar.bz2',
 '85c83ca46fad11c71e4749042d3e4d0cabf21238966c3c41924016d9c53f25cf':
 'EyesJapanDataset_smplx_g.tar.bz2'}

def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def restore(index,archive_dir,output,limit=0):
    rows=json.loads(pathlib.Path(index).read_text(encoding='utf-8'))
    if limit:rows=rows[:limit]
    out=pathlib.Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    needed={};done=[]
    for row in rows:
        name=pathlib.Path(row['local_raw_path']).name
        assert name.endswith('_stageii.npz') and '/' not in name and '\\' not in name
        dest=out/name
        assert dest.parent==out and not dest.is_symlink()
        if dest.exists():assert sha(dest)==row['raw_sha256'];done.append(row);continue
        needed.setdefault(row['archive_sha256'],{})[row['archive_member']]=row
    for digest,members in needed.items():
        archive=pathlib.Path(archive_dir)/ARCHIVES[digest]
        assert sha(archive)==digest,'Archive checksum differs'
        with tarfile.open(archive,mode='r|bz2') as tar:
            for member in tar:
                if member.name not in members:continue
                assert member.isfile()
                row=members[member.name];dest=out/pathlib.Path(row['local_raw_path']).name
                assert shutil.disk_usage(out).free>member.size+6*1024**3,'6GiB reserve required'
                partial=dest.with_suffix(dest.suffix+'.restore.partial')
                h=hashlib.sha256()
                with tar.extractfile(member) as source,partial.open('xb') as target:
                    for block in iter(lambda:source.read(8*1024**2),b''):h.update(block);target.write(block)
                assert h.hexdigest()==row['raw_sha256'] and partial.stat().st_size==member.size
                partial.rename(dest);done.append(row);del members[member.name]
                if not members:break
        assert not members,'Indexed member missing from original archive'
    assert len(done)==len(rows)
    return {'restored_or_verified':len(done),'all_member_sha256_match':True,'overwrite_performed':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--archive-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();print(json.dumps(restore(a.index,a.archive_dir,a.output,a.limit)))
