"""Public small evidence only; no motion arrays, checkpoints or contact records."""
from pathlib import Path
import json
import tarfile
import time
from verify_execution_lineage_v39_20261012 import sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/support_reference_probe_v63_20261012'
DEST=BASE/'public_exports';DEST.mkdir(exist_ok=True)
files=[];excluded=[]
for name in ('support_reference_probe_v62_20261012','support_reference_probe_v63_20261012','native_manifold_analysis_v57_20261012'):
    directory=ROOT/'results'/name
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or DEST in p.parents:continue
        if p.suffix not in ('.json','.md','.log','.txt') or p.name=='native_contacts.json' or p.stat().st_size>2*1024**2:
            excluded.append(str(p.relative_to(ROOT)));continue
        if 'raw' in p.relative_to(directory).parts and p.name not in ('metadata.json','metrics.json','receipt.json','lineage.json'):
            excluded.append(str(p.relative_to(ROOT)));continue
        files.append(p)
stamp=time.time_ns();manifest=DEST/('public_export_%d.json'%stamp)
manifest.write_text(json.dumps(dict(files={str(p.relative_to(ROOT)):sha(p) for p in files},excluded=excluded,
    source_sha256=sha(Path(__file__)),licensed_arrays_exported=False,simulation_calls=0,generation_calls=0),indent=2))
archive=DEST/('support_probe_public_%d.tar.gz'%stamp)
with tarfile.open(archive,'w:gz') as t:
    for p in files+[manifest]:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print('SUPPORT_PROBE_PUBLIC_EXPORT',archive,len(files),sha(archive),flush=True)
