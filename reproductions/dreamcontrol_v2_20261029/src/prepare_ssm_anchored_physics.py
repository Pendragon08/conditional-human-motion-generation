"""Stage two unaccepted SSM candidates for existing static MuJoCo diagnostics."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SRC=ROOT/'ssm_anchored_20hz_candidates_20261031'
PREVIOUS=ROOT/'eight_static_robot_quality_20261031'
HERE=Path(__file__).resolve().parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p,v):
    with p.open('x') as f: json.dump(v,f,indent=2); f.write('\n')


def main():
    audit=json.loads((SRC/'independent_audit.json').read_text())
    expected={x['id']:x['output_sha256'] for x in audit['cases']}
    assert set(expected)=={'007410','011120'}
    records=[]; states=[]
    for sid in ('007410','011120'):
        path=SRC/(sid+'.npz'); assert sha(path)==expected[sid]
        target=HERE/(sid+'.npz')
        with path.open('rb') as src,target.open('xb') as dst: shutil.copyfileobj(src,dst)
        assert sha(target)==expected[sid]
        records.append({'sample_id':sid,'split':'unassigned_review_only','feature_sha256':expected[sid]})
        states.append({'sample_id':sid,'state_path':target.name,'state_sha256':expected[sid], 'feature_sha256':expected[sid]})
    write_json(HERE/'manifest.json',{'records':records})
    write_json(HERE/'state_manifest.json',{'records':states})
    write_json(HERE/'build_report.json',{'manifest_sha256':sha(HERE/'manifest.json'),
                                      'state_manifest_sha256':sha(HERE/'state_manifest.json')})
    old=json.loads((PREVIOUS/'report.json').read_text())
    src=PREVIOUS/'measure_eight_static_robot_quality.py'
    assert sha(src)==old['script_sha256']
    text=src.read_text()
    assert text.count("INPUT = ROOT / 'eight_projected_preflight_20261031'")==1
    assert text.count('len(records) != 8 or len(states) != 8')==1
    text=text.replace("INPUT = ROOT / 'eight_projected_preflight_20261031'",'INPUT = Path(__file__).resolve().parent')
    text=text.replace('len(records) != 8 or len(states) != 8','len(records) != 2 or len(states) != 2')
    with (HERE/'measure.py').open('x') as f:f.write(text)
    audit_src=PREVIOUS/'audit_eight_static_robot_quality.py'
    text=audit_src.read_text()
    assert text.count("INPUT = ROOT / 'eight_projected_preflight_20261031'")==1
    text=text.replace("INPUT = ROOT / 'eight_projected_preflight_20261031'",'INPUT = Path(__file__).resolve().parent')
    with (HERE/'audit.py').open('x') as f:f.write(text)
    write_json(HERE/'prepare.json',{'candidate_audit_sha256':sha(SRC/'independent_audit.json'),
        'previous_measure_script_sha256':sha(src),'previous_audit_script_sha256':sha(audit_src),
        'measure_sha256':sha(HERE/'measure.py'),'audit_sha256':sha(HERE/'audit.py'),
        'training_ready':False,'note':'Static assumed-plane and open-hand diagnostics, no physical acceptance.'})


if __name__=='__main__':main()
