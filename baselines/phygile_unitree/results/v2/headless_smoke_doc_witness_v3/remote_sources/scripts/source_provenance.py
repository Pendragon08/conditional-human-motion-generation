"""Bind actual imported project sources and explicit inputs before stepping."""
from pathlib import Path
import hashlib,importlib.metadata,json,sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def capture(output,roots,inputs=()):
    roots=[Path(p).resolve() for p in roots if Path(p).exists()]
    files={}
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if not value:continue
        path=Path(value).resolve()
        if path.suffix not in {'.py','.sh'} or not path.is_file():continue
        if any(path==r or r in path.parents for r in roots):files[str(path)]=sha(path)
    for value in inputs:
        path=Path(value).resolve()
        if path.is_file():files[str(path)]=sha(path)
    versions={}
    for name in ['torch','rsl-rl-lib','numpy','gymnasium','mjlab','mujoco','mujoco-warp','warp-lang']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    manifest={'python':sys.version,'executable':sys.executable,'files':dict(sorted(files.items())),'package_versions':versions,
      'scope':'Actual loaded project/framework Python files and explicit input files; package versions identify other installed binary dependencies.'}
    path=Path(output);path.write_bytes((json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode())
    return {'manifest':str(path),'sha256':sha(path),'source_files':len(files)}

def verify(proof):
    path=Path(proof['manifest'])
    if sha(path)!=proof['sha256']:raise RuntimeError('Source manifest changed during execution')
    for name,digest in json.loads(path.read_text(encoding='utf-8'))['files'].items():
        if sha(name)!=digest:raise RuntimeError('Executed source/input changed during run: '+name)
    return True
