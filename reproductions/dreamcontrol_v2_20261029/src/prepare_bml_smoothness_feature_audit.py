"""Adapt the existing independent feature auditor to fixed-smoothness input."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD=ROOT/'bml_global_features_20261031'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    src=OLD/'audit.py'
    text=src.read_text()
    a="ROOT/'bml_global_candidates_20261031'/label";b="solve/'audit_report.json'"
    if text.count(a)!=1 or text.count(b)!=1:raise ValueError('unexpected auditor')
    text=text.replace(a,"ROOT/'bml_joint_smoothness_20261031'/label")
    text=text.replace(b,"solve/'audit_report_v2.json'")
    with (HERE/'audit.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
    with (HERE/'audit_prepare_manifest.json').open('x',encoding='utf8') as f:
        json.dump({'original_sha256':sha(src),'adapted_sha256':sha(HERE/'audit.py')},f,indent=2);f.write('\n')


if __name__=='__main__':main()
