"""Finalize only completed, supervised real contact-consistent filtering exports."""
import argparse,collections,hashlib,json,os,pathlib,re
from apply_tracking_filter_v3 import apply
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
assert os.getuid()==1013 and re.fullmatch('[a-z0-9_]+',a.run);os.sched_setaffinity(0,{0,1,2,3})
out=ROOT/'tracker_validation'/a.run;assert out.resolve()==out and out.stat().st_uid==1013
receipt=out/'receipt.json';exitpath=ROOT/'receipts'/('strict_contact_'+a.run+'_exit.json')
assert exitpath.exists(),'Actual supervisor exit required, not a live exporter'
r=json.loads(receipt.read_text());e=json.loads(exitpath.read_text())
assert r['status']=='TEXTOP_TWO_ROUND_FILTER_EXPORT_FINISHED' and r['output_writers_closed'] is True
assert e['child_exit_code']==0 and e['timed_out'] is False
output_paths=[out/'evidence_v3.private.json',out/'decisions_v3.private.json',out/'post_exit_filter_receipt.json']
assert not any(x.exists() for x in output_paths),'Prior finalized evidence must be preserved'
evidence=json.loads((out/'evidence_inputs.private.json').read_text())
for key,path in [('evaluation_receipt',receipt),('evaluation_exit_receipt',exitpath)]:evidence[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
decisions=apply(evidence);counts=dict(collections.Counter(x['decision'] for x in decisions))
for path,data in zip(output_paths[:2],[evidence,{'schema':'textop.dynamic_filter.v3','decisions':decisions,'dataset_accepted':False,'corpus_acceptance_pending':True}]):
 with path.open('x') as f:f.write(json.dumps(data,indent=2))
summary={'status':'TWO_ROUND_DECISIONS_VERIFIED_CORPUS_PENDING','schema':'textop.dynamic_filter.v3','reference_sources':len(decisions),'decision_counts':counts,'physical_filter_executed':True,'dataset_accepted':False,'full_generator_training_started':False,'policy_checkpoint_sha256':evidence['policy_checkpoint']['sha256'],'reference_fingerprint':r['reference_fingerprint'],'contact_profile_sha256':r['contact_profile_sha256'],'private_evidence_sha256':hashlib.sha256(output_paths[0].read_bytes()).hexdigest(),'private_decisions_sha256':hashlib.sha256(output_paths[1].read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
with output_paths[2].open('x') as f:f.write(json.dumps(summary,indent=2))
print(json.dumps(summary))
