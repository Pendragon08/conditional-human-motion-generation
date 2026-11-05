import hashlib,json,pathlib
import numpy as np
R=pathlib.Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');B=R/'stage3_supported_references_20261104';W=B/'bulk_v1';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert not (W/'library_records.jsonl').exists();refs=[];proofs=[]
for directory in [B/'native_pilot_v1/pour',B/'native_pilot_v1/bimanual_box_lift',B/'native_push_pilot_v1/box_push_pull']:
 report=json.loads((directory/'report.json').read_text());assert report['admitted_references']==2 and report['candidates']==2
 bank=json.loads((directory/'reference_bank.json').read_text());assert len(bank['references'])==2
 for ref in bank['references']:
  proof=json.loads(pathlib.Path(ref['native_proof']).read_text());r=json.loads(pathlib.Path(ref['kinematic_report']).read_text())
  assert proof['eligible_for_calibrated_scene_reference'] and all(proof['native_checks'].values()) and r['passed_requested_kinematic_checks']
  assert sha(ref['native_proof'])==ref['native_proof_sha256'] and sha(ref['path'])==ref['sha256']==proof['source_state_sha256']
  assert sha(ref['kinematic_report'])==proof['kinematic_report_sha256'] and sha(r['sample'])==r['sample_sha256']
  assert proof['native_replay_frames']==180 and proof['source_fps']==20 and proof['asset_sha256']=='13b283a83a2b250a3846d9950e0b641c959e3ce80fcaf706ea060ce0d8fb7888'
  with np.load(ref['path'],allow_pickle=False) as z:fp=hashlib.sha256(b''.join(np.ascontiguousarray(z[k]).tobytes() for k in ['q','root_pos','root_quat','hand_closed'])).hexdigest()
  with np.load(r['sample'],allow_pickle=False) as z:m=json.loads(z['metadata_utf8'].tobytes())
  refs.append({**ref,'state_fingerprint':fp,'initial_noise_sha256':m['initial_noise_sha256'],'feature_content_sha256':m['feature_content_sha256']})
 proofs.append({'report':str(directory/'report.json'),'sha256':sha(directory/'report.json')})
for key in ['candidate_id','state_fingerprint','initial_noise_sha256','feature_content_sha256']:assert len({r[key] for r in refs})==6
(W/'library_records.jsonl').open('x').write(''.join(json.dumps(r)+'\n' for r in refs))
report={'status':'six_native_pilots_independent_binding_passed','native_admitted_counts':{t:sum(r['task']==t for r in refs) for t in ['pour','bimanual_box_lift','box_push_pull']},'proofs':proofs,'formal_training_started':False,'scope':'reference geometry applicability; not dynamic task execution'}
(W/'pilot_admission_audit.json').open('x').write(json.dumps(report,indent=2));print(json.dumps(report))
