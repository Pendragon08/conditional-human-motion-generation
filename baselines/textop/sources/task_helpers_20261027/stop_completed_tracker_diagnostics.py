import pathlib,os,json,signal,time,subprocess
assert os.getuid()==1013
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
entries=[]
for pid,name in [(3412830,'author_v1'),(3414800,'document_v1')]:
 proc=pathlib.Path('/proc')/str(pid)
 receipt=json.loads((r/'tracker_validation'/name/'receipt.json').read_text())
 assert receipt['status']=='TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED' and receipt['optimizer_updates']==0
 if not proc.exists():continue
 assert proc.stat().st_uid==1013
 cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 assert '/scripts/tracker_kernel_witness.py' in cmd and '/tracker_validation/'+name in cmd
 entries.append({'pid':pid,'run':name,'command':cmd,'reason':'Numerical diagnostic complete; SimulationApp.close default replicator wait did not return','numerical_receipt':receipt,'signals':[]})
for x in entries:os.kill(x['pid'],signal.SIGTERM);x['signals'].append('SIGTERM')
time.sleep(2)
for x in entries:
 proc=pathlib.Path('/proc')/str(x['pid'])
 if proc.exists() and (proc/'cmdline').read_bytes():
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  assert '/scripts/tracker_kernel_witness.py' in cmd and '/tracker_validation/'+x['run'] in cmd
  os.kill(x['pid'],signal.SIGKILL);x['signals'].append('SIGKILL')
proof={'scope':'two newly launched zero-update diagnostic children only','entries':entries,'existing_GMR_process_retained':pathlib.Path('/proc/3319567').exists(),'paused_57D_process_retained':pathlib.Path('/proc/3237858').exists()}
(r/'tracker_validation/teardown_v1_audit.json').write_text(json.dumps(proof,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='entries'}))
