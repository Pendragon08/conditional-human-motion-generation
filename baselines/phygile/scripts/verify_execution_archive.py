from pathlib import Path
import hashlib,json,tarfile
root=Path(__file__).resolve().parents[1]
out=root/'results/continuation_2000'
archive=root/'release_assets/phygile_continuation_execution_source.tar.gz'
with tarfile.open(archive,'r:gz') as tar:
    hashes={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers() if m.isfile()}
checks=[]
train=json.loads((out/'train/receipt.json').read_text())
names=[n for n in train['code_sha256'] if n.startswith('upstream/')]
names += ['scripts/resume_training.py','scripts/train_extended_2000.sh','scripts/run_isaac.sh','configs/moe_pilot.yaml']
for name in names:
    assert hashes[name]==train['code_sha256'][name],name
checks.append({'stage':'training','matched_files':len(names)})
suite=json.loads((out/'evaluation_suite.json').read_text())
for entry in suite['entries']:
    receipt=json.loads((out/'evaluation_logs'/entry['name']/'receipt.json').read_text())
    for name in ['scripts/evaluate_pilot.py','scripts/run_isaac.sh','configs/moe_pilot.yaml']:
        assert hashes[name]==receipt['code_sha256'][name],name
checks.append({'stage':'evaluation','matched_receipts':len(suite['entries'])})
receipt=json.loads((out/'state_comparison_check/receipt.json').read_text())
for name in ['scripts/verify_continuation_states.py','scripts/run_isaac.sh']:
    assert hashes[name]==receipt['code_sha256'][name],name
checks.append({'stage':'final_state_verification','matched_files':2})
core=json.loads((root/'results/reimplementation_core/verification.json').read_text())
for name,digest in core['source_sha256'].items():assert hashes['reimplementation/'+name]==digest,name
checks.append({'stage':'generator_core_numerical_checks','matched_files':len(core['source_sha256'])})
record={'status':'PASSED','archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'checks':checks,'scope':'Server execution source. External simulator and robot-description dependency excluded and documented separately. Local report builder remains versioned in Git.'}
(out/'execution_archive_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
