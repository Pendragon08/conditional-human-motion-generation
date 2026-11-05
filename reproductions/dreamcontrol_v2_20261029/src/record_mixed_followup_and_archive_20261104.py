import json,pathlib,hashlib
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104'
active=B/'active_pipeline.json';j=json.loads(active.read_text())
launch=B/'mixed2000_completion_v1_launch.json'
j['mixed2000_completion_followup']={'launch':str(launch),'status':str(B/'mixed2000_completion_v1/status.json'),'pid':json.loads(launch.read_text())['pid'],'training_started':False}
tmp=active.with_suffix('.followup.tmp');tmp.write_text(json.dumps(j,indent=2));tmp.replace(active)
for version,reason in [('v1','first same-UID /proc FD scan stopped on PermissionError; no checkpoint copied or removed'),('v2','bounded retry still could not read foreign /proc FD tables; no checkpoint copied or removed')]:
 d=pathlib.Path('/home/ouyanghao/dreamcontrol_v2_cold_archive_20261104')/('retired_sm_style_v2_'+version)
 p=d/'failed_preserved.json'
 if not p.exists():
  assert list(d.iterdir())==[]
  p.write_text(json.dumps({'status':'failed_preserved_no_source_mutation','reason':reason,'followup':'v3 used Linux kernel exclusive file leases plus complete verified gzip copies; complete index under project checkpoint_archival_20261104'},indent=2))
index=R/'checkpoint_archival_20261104/retired_sm_v2_manifest.json';m=json.loads(index.read_text())
assert m['status']=='retired_sm_snapshots_losslessly_cold_archived' and m['freed_data_bytes']==1143614108
for row in m['records']:
 assert not pathlib.Path(row['source']).exists() and pathlib.Path(row['archive']).exists()
 assert pathlib.Path(row['source']+'.archived.json').exists()
print(json.dumps({'bulk_status':json.loads((B/'bulk_v1/controller_status.json').read_text())['status'],'final_followup':json.loads((B/'mixed2000_completion_v1/status.json').read_text())['status'],'archive_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest()}))
