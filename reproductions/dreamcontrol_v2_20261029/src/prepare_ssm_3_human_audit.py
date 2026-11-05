"""Adapt the prior audited reconstruction checker to a new script filename."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
OLD = ROOT / 'humanml_mirror_time_witness_v2_20261030'
HERE = Path(__file__).resolve().parent


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior = json.loads((OLD / 'independent_audit.json').read_text())
    source = OLD / 'audit_ssm_mirror_time_witness.py'
    if sha(source) != prior['script_sha256']: raise ValueError('prior audit changed')
    text = source.read_text(encoding='utf8')
    old = "OUT/'ssm_joint_mirror_time_witness.py'"
    if text.count(old) != 1: raise ValueError('unexpected audit source')
    target = HERE / 'audit.py'
    with target.open('x', encoding='utf8', newline='\n') as f:
        f.write(text.replace(old, "OUT/'reconstruct.py'"))
    with (HERE / 'audit_prepare.json').open('x', encoding='utf8') as f:
        json.dump({'source_sha256': sha(source), 'adapted_sha256': sha(target),
                   'change': 'reconstruction script filename only'}, f, indent=2); f.write('\n')


if __name__ == '__main__': main()
