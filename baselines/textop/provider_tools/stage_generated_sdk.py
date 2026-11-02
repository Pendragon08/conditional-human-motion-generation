"""Stage separate generated-reference SDK validation after real neural witnesses."""
import ast,base64,json,pathlib
from session import remote
B=pathlib.Path(__file__).resolve().parents[1]
names=['scripts/generated_pipeline_protocol.py','scripts/run_generated_pipeline_sdk.py','scripts/supervise_generated_pipeline.py']
for n in names:ast.parse((B/n).read_text(encoding='utf-8'))
payload={n:base64.b64encode((B/n).read_bytes()).decode() for n in names}
script=r'''
import ast,base64,fcntl,hashlib,json,os,pathlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert os.getuid()==1013;lk=(r/'receipts/full_library_capacity.lock').open('a+');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=r/'receipts/generated_sdk_stage_v1.json';assert not out.exists();witnesses=[]
for name in ['generated_author_v1','generated_document_v1']:
 rp=r/'generated_reference_private'/name/'receipt.json';ep=r/'receipts'/('quick_generation_'+name+'_exit.json');j=json.loads(rp.read_text());ex=json.loads(ep.read_text())
 assert ex['supervision_passed'] and ex['child_exit_code']==0 and ex['evaluation_receipt_sha256']==sha(rp) and j['status']=='TRAINED_QUICK_TEXT_REFERENCE_GENERATED' and j['output_writers_closed']
 assert j['spec_sha256']==sha(r/'configs/quick-generated-reference-spec.json') and sha(j['artifact']['file'])==j['artifact']['sha256']
 witnesses.append({'run':name,'receipt_sha256':sha(rp),'exit_sha256':sha(ep)})
author=json.loads((r/'generated_reference_private/generated_author_v1/receipt.json').read_text())
payload=PAYLOAD
for n,enc in payload.items():
 p=r/n;assert not p.exists();raw=base64.b64decode(enc);ast.parse(raw.decode());p.write_bytes(raw)
bound=['scripts/generated_pipeline_protocol.py','scripts/run_generated_pipeline_sdk.py','scripts/quick_pilot_protocol.py','scripts/quick_pilot_trace.py','scripts/strict_tracking_coverage.py','scripts/paper_filter_config.py','scripts/validated_contact_profile.py','scripts/contact_process_supervision.py','scripts/contact_filter_resources.py','scripts/run_tracker.sh','scripts/bounded_headless_close.py','configs/tracker-env-spec.json','configs/quick-pilot-spec.json','configs/filter-training-plan.json']
spec={'schema':'textop.generated_execution_validation.v1','validation_enabled':True,'source_kind':'trained_generated_reference','full_paper_or_realtime_acceptance':False,
 'generation_spec_sha256':sha(r/'configs/quick-generated-reference-spec.json'),'generation_witnesses':witnesses,
 'input':{'file':author['artifact']['file'],'sha256':author['artifact']['sha256'],'private_metadata_sha256':author['private_metadata_sha256']},
 'executor_sha256':sha(r/'scripts/run_generated_pipeline_sdk.py'),'supervisor_sha256':sha(r/'scripts/supervise_generated_pipeline.py'),'bound_sources':{n:sha(r/n) for n in bound},
 'scope':'One generated258pose/257interval stream, same29DoF trackerURDF FK and5kprivilegedteacher strict physics, preserve failures. Offline model-generation→execution prototype, not real-time closedloop/WBC or source-selected8278 filtering'}
sp=r/'configs/quick-generated-sdk-spec.json';assert not sp.exists();sp.write_text(json.dumps(spec,indent=2)+'\n')
proof={'status':'GENERATED_SDK_VALIDATION_PROFILE_STAGED','spec_sha256':sha(sp),'spec_base64':base64.b64encode(sp.read_bytes()).decode(),'generation_witnesses':witnesses,'new_sdk_dispatches':0,'new_optimizer_updates':0}
out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''.replace('PAYLOAD',repr(payload))
j=json.loads(remote(script,timeout=60));(B/'configs/quick-generated-sdk-spec.json').write_bytes(base64.b64decode(j['spec_base64']));out=B/'results/generated_execution_20261026';out.mkdir(exist_ok=True);(out/'stage_v1.json').write_text(json.dumps(j,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in j.items() if k!='spec_base64'}))
