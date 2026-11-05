"""Prepare an isolated feature exporter for the fixed-smoothness solve."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD=ROOT/'bml_global_features_20261031'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    previous=json.loads((OLD/'S08_Novice/report.json').read_text())
    source=OLD/'export.py'
    if sha(source)!=previous['script_sha256']:raise ValueError('source exporter changed')
    text=source.read_text()
    old="ROOT/'bml_global_candidates_20261031'/label"
    if text.count(old)!=1:raise ValueError('unexpected path in exporter')
    text=text.replace(old,"ROOT/'bml_joint_smoothness_20261031'/label")
    old="solve/'audit_report.json'"
    if text.count(old)!=2:raise ValueError('unexpected audit reference')
    text=text.replace(old,"solve/'audit_report_v2.json'")
    with (HERE/'export.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
    for name in ('feature263.py','g1_fk.py'):
        with (HERE/name).open('xb') as f:f.write((OLD/name).read_bytes())
    out={'source_sha256':sha(source),'adapted_sha256':sha(HERE/'export.py'),
         'feature263_sha256':sha(HERE/'feature263.py'),'g1_fk_sha256':sha(HERE/'g1_fk.py'),
         'changes':['solve directory','audited report filename'],'training_ready':False}
    with (HERE/'prepare_manifest.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
