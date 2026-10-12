"""Separate version after preflight missing-key failure; preserve original attempt."""
import argparse
import json
import subprocess
import controller_state_diagnosis_20261012 as prior

ROOT=prior.ROOT
OUT=ROOT/'results/controller_state_diagnosis_v2_20261012'
DOC=ROOT/'docs/controller_state_diagnosis_v2_20261012.md'
PIN='d8950bdb78312fcd2844f336b3fc079e7368c87f'
prior.OUT=OUT

def main(mode):
    if mode!='prepare':prior.main(mode);return
    failure=prior.read(ROOT/'results/controller_state_diagnosis_20261012/DOCUMENT_WITNESS.json')
    assert failure['status']=='FAIL', 'Preserve and bind actual original failed witness'
    actual=subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'rev-parse','HEAD'],text=True).strip()
    assert actual==PIN
    assert subprocess.check_output(['git','-C',str(ROOT/'third_party/CLoSD'),'diff','--name-only','HEAD','--','closd'],text=True)==''
    prior.prepare()
    version=prior.read(OUT/'versions.json')
    version.update(upstream_commit=actual,original_failed_witness_sha256=prior.sha(ROOT/'results/controller_state_diagnosis_20261012/DOCUMENT_WITNESS.json'))
    for path in (ROOT/'scripts/controller_state_diagnosis_v2_20261012.py',DOC):
        version['files'][str(path.relative_to(ROOT))]=prior.sha(path)
    # This is CPU-only preparation before any request, witness or GPU execution.
    (OUT/'versions.json').write_text(json.dumps(version,indent=2),encoding='utf8')
    prior.validate()
    print('CONTROLLER_STATE_V2_PREPARED_BOUND_UPSTREAM',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for mode in ('prepare','check','run'):g.add_argument('--'+mode,action='store_true')
    args=p.parse_args();main(next(m for m in ('prepare','check','run') if getattr(args,m)))
