"""New generated-reference provenance; never pretends to be an AMASS cohort."""
import hashlib,json,pathlib
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()

def gate(root,spec):
 assert spec['schema']=='textop.generated_execution_validation.v1' and spec['validation_enabled']
 assert spec['source_kind']=='trained_generated_reference' and spec['full_paper_or_realtime_acceptance'] is False
 for name,h in spec['bound_sources'].items():assert sha(root/name)==h
 gp=root/'configs/quick-generated-reference-spec.json';assert sha(gp)==spec['generation_spec_sha256']
 for w in spec['generation_witnesses']:
  rp=root/'generated_reference_private'/w['run']/'receipt.json';ep=root/'receipts'/('quick_generation_'+w['run']+'_exit.json');r=json.loads(rp.read_text());ex=json.loads(ep.read_text())
  assert sha(rp)==w['receipt_sha256'] and sha(ep)==w['exit_sha256'] and ex['evaluation_receipt_sha256']==sha(rp)
  assert ex['supervision_passed'] and ex['child_exit_code']==0 and r['output_writers_closed'] and r['status']=='TRAINED_QUICK_TEXT_REFERENCE_GENERATED'
  assert r['frames']==258 and r['optimizer_updates']==0 and r['physics_steps']==0 and r['posterior_future_used'] is False
 assert len(spec['generation_witnesses'])==2
 item=spec['input'];p=pathlib.Path(item['file']);assert sha(p)==item['sha256'] and sha(p.parent/'generation.private.json')==item['private_metadata_sha256']
 return item
