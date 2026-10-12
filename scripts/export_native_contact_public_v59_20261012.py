"""Small public records only; never export licensed trajectory/contact arrays."""
import json
from pathlib import Path
import tarfile
import time
from verify_execution_lineage_v39_20261012 import sha

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/native_manifold_v56_20261012/public_exports'
directories=[ROOT/'results'/name for name in ('paired_feedback_analysis_v54_20261012',
    'selected_construction_binding_v55_20261012','native_manifold_v56_20261012','native_manifold_analysis_v57_20261012')]
directories.append(ROOT/'.aris/traces/experiment-audit/2026-10-12_run01')
DEST.mkdir(exist_ok=True)
files=[];excluded=[]
for directory in directories:
    if not directory.exists():continue
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or DEST in p.parents:continue
        if p.suffix not in ('.json','.md','.log','.txt') or p.name=='native_contacts.json' or p.stat().st_size>2*1024**2:
            excluded.append(str(p.relative_to(ROOT)));continue
        if 'raw' in p.relative_to(directory).parts and p.name not in ('metadata.json','metrics.json','receipt.json','lineage.json'):
            excluded.append(str(p.relative_to(ROOT)));continue
        files.append(p)
manifest=dict(source_sha256=sha(Path(__file__)),files={str(p.relative_to(ROOT)):sha(p) for p in files},
    excluded=excluded,licensed_arrays_exported=False,simulation_calls=0,generation_calls=0,time=time.time())
stamp=time.time_ns();m=DEST/('public_export_%d.json'%stamp)
m.write_text(json.dumps(manifest,indent=2));archive=DEST/('native_contact_public_%d.tar.gz'%stamp)
with tarfile.open(archive,'w:gz') as tar:
    for p in files+[m]:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print('NATIVE_CONTACT_PUBLIC_EXPORT',archive,len(files),sha(archive),flush=True)
