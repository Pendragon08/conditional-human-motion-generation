import json, hashlib, subprocess
from pathlib import Path
root=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation')
out=root/'results/controller_state_diagnosis_20261012'
def sha(p):
    if not p.is_file(): return None
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(p.read_text()) if p.is_file() else None
v=read(out/'versions.json')
paths=['runtime_witness.json','versions.json','budget_ledger.json','preparation.json','state.json','multitask.log','multitask_parent_return.json','requests/multitask.json','raw/multitask/metadata.json','raw/multitask/metrics.json','raw/multitask/trajectories.npz','raw/multitask/initial_states.npz','raw/multitask/lineage.json','raw/multitask/receipt.json','raw/multitask/loaded_input_binding.json','raw/multitask/loaded_checkpoint_binding.json']
evidence=dict(output_presence={p:(out/p).exists() for p in paths},output_sha256={p:sha(out/p) for p in paths},raw_directories=[p.name for p in (out/'raw').iterdir()],versions_keys=sorted(v),upstream_commit_field_present='upstream_commit' in v,actual_upstream_commit=subprocess.check_output(['git','-C',str(root/'third_party/CLoSD'),'rev-parse','HEAD'],text=True).strip(),tracked_closd_diff=subprocess.check_output(['git','-C',str(root/'third_party/CLoSD'),'diff','--name-only','HEAD','--','closd'],text=True),budget_ledger=read(out/'budget_ledger.json'),state=read(out/'state.json'),preparation=read(out/'preparation.json'),source_sha256={p:sha(root/p) for p in ['docs/controller_state_diagnosis_20261012.md','scripts/controller_state_diagnosis_20261012.py','scripts/controller_state_collect_20261012.py','scripts/reference_child_binding_collect_20261012.py','scripts/child_input_guard_20261012.py']})
print(json.dumps(evidence,indent=2))
