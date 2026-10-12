"""Schema-only completion of separate addendum. No numerical rerun."""
import json,hashlib
from pathlib import Path
OUT=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=OUT/'NUMERICAL_CONTRACT_ADDENDUM.json';r=json.loads(p.read_text())
assert r['status']=='PASS' and r['all_input_hashes_unchanged']
assert sha(OUT/'DOCUMENT_WITNESS.json')==r['original_FAIL_witness_sha256']
assert sha(OUT/'runtime_witness.json')==r['runtime_witness_sha256']
r.update(original_witness_sha256=r['original_FAIL_witness_sha256'],versions_sha256=r['input_hashes']['versions.json'],simulation_rerun=False,source_changed=False,experimental_thresholds_changed=False)
r['commands'] += [
    {'type':'upload_addendum_audit','command':'scp -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 D:/具身智能/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/numerical_contract_addendum_audit.py ouyanghao@10.147.17.32:/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/numerical_contract_addendum_audit.py'},
    {'type':'schema_only_finalize','command':'ssh -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32 /data/code/ouyanghao/oyh/humanoid-combat-motion/.venv-pbhc-training/bin/python /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/finalize_numerical_addendum_schema.py','script_sha256':sha(Path(__file__))},
    {'type':'sync_final_reports','command_template':'scp -i C:/Users/Pende/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ouyanghao@10.147.17.32:/data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/<report> D:/具身智能/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/<report>','reports':['NUMERICAL_CONTRACT_ADDENDUM.json','NUMERICAL_CONTRACT_ADDENDUM.md','numerical_adjudication_budget_ledger.json']},
    {'type':'remote_byte_hash_verification','command':'sha256sum /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/NUMERICAL_CONTRACT_ADDENDUM.json /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/NUMERICAL_CONTRACT_ADDENDUM.md /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/numerical_adjudication_budget_ledger.json /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/DOCUMENT_WITNESS.json /data/code/ouyanghao/oyh/conditional-human-motion-generation/results/support_reference_probe_v63_20261012/DOCUMENT_WITNESS.md'}]
p.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf8')
md=OUT/'NUMERICAL_CONTRACT_ADDENDUM.md'
md.write_text(md.read_text()+'\nMachine-readable bindings: '+json.dumps({k:r[k] for k in ('status','original_witness_sha256','runtime_witness_sha256','versions_sha256','simulation_rerun','source_changed','experimental_thresholds_changed')})+'\n',encoding='utf8')
print(json.dumps({'status':r['status'],'json_sha256':sha(p),'md_sha256':sha(md),'original_witness_unchanged':sha(OUT/'DOCUMENT_WITNESS.json')==r['original_witness_sha256'],'schema_fields':{k:r[k] for k in ('original_witness_sha256','runtime_witness_sha256','versions_sha256','simulation_rerun','source_changed','experimental_thresholds_changed')}},indent=2))
