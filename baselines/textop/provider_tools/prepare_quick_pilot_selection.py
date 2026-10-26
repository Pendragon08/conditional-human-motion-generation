"""Check the new selector against actual private metadata; no physics or rejection."""
import base64, hashlib, json, pathlib
from session import remote
BASE=pathlib.Path(__file__).resolve().parents[1]
out=BASE/'results/quick_pilot_candidate_20261026'
assert out.exists() and not (out/'selection_metadata.json').exists()
private=pathlib.Path('D:/具身智能/TextOp复现_20261024/datasets/quick_pilot_private_20261026')
private.mkdir(exist_ok=False)
script='''import base64,collections,hashlib,json,pathlib,sys
root=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sys.path.insert(0,str(root/'scripts'))
from quick_pilot_protocol import select_sources
spec=json.loads((root/'configs/quick-pilot-spec.json').read_text());plan=json.loads((root/'configs/filter-training-plan.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'scripts/quick_pilot_protocol.py')==spec['protocol_sha256']
rows=[]
for name in plan['input_reference_exports']:
 folder=pathlib.Path('/dev/shm/ouyanghao_textop_20261025/tracker_references')/name
 receipt=json.loads((folder/'receipt.json').read_text());p=folder/'manifest.private.jsonl'
 assert sha(p)==receipt['private_manifest_sha256']
 rows.extend(r for r in map(json.loads,p.read_text().splitlines()) if not r['mirrored'])
fingerprint=lambda items:hashlib.sha256(json.dumps([dict(sha256=r['sha256'],frames=r['frames']) for r in items],sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert len(rows)==8278 and fingerprint(rows)==spec['expected_reference_fingerprint']
selected={stage:select_sources(rows,stage,spec) for stage in ['smoke','sample']}
dest=root/'tracker_validation/quick_pilot_selection_metadata_v1';dest.mkdir(exist_ok=False)
private=dict(quality_label=spec['quality_label'],protocol_sha256=spec['protocol_sha256'],
 reference_fingerprint=fingerprint(rows),stages={stage:[dict(global_index=i,canonical_source=rows[i]['canonical_source'],
 frames=rows[i]['frames'],split=rows[i]['split'],reference_sha256=rows[i]['sha256']) for i in indices] for stage,indices in selected.items()})
path=dest/'selection.private.json';path.write_text(json.dumps(private,indent=2))
summary=dict(status='QUICK_PILOT_ACTUAL_METADATA_SELECTION_VERIFIED_PHYSICS_PENDING',
 reference_sources=8278,reference_fingerprint=fingerprint(rows),selected={})
for stage,indices in selected.items():
 groups=collections.Counter(pathlib.PurePosixPath(rows[i]['canonical_source'].replace('\\\\','/')).parts[0] for i in indices)
 bins=collections.Counter('50..250' if rows[i]['frames']<=250 else '251..750' if rows[i]['frames']<=750 else '751..2000' for i in indices)
 summary['selected'][stage]=dict(count=len(indices),minimum_frames=min(rows[i]['frames'] for i in indices),maximum_frames=max(rows[i]['frames'] for i in indices),
 dataset_counts=dict(groups),duration_bin_counts=dict(bins),split_counts=dict(collections.Counter(rows[i]['split'] for i in indices)),
 selected_order_fingerprint=fingerprint([rows[i] for i in indices]),unattempted_sources=8278-len(indices))
summary.update(smoke_sample_overlap=len(set(selected['smoke'])&set(selected['sample'])),
 excluded_by_sample_duration_scope=sum(not 50<=r['frames']<=2000 for r in rows),
 failure_statistics_used=False,sources_rejected=0,physical_trials=0,sdk_runs=0,new_optimizer_updates=0,
 private_selection_sha256=sha(path),private_selection_bytes=path.stat().st_size)
(dest/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(summary=summary,private_b64=base64.b64encode(path.read_bytes()).decode())))
'''
try:
 result=json.loads(remote(script,timeout=60));raw=base64.b64decode(result.pop('private_b64'))
 assert hashlib.sha256(raw).hexdigest()==result['summary']['private_selection_sha256']
 (private/'selection.private.json').write_bytes(raw)
 (out/'selection_metadata.json').write_text(json.dumps(result['summary'],indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result['summary']))
except BaseException as error:
 (out/'selection_failure.json').write_text(json.dumps(dict(error=repr(error)),indent=2),encoding='utf-8');raise
