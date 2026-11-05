"""Adapt a checked window-audit script to 355 full-rate source frames."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    src=ROOT/'pyroki_ssm_000008_human_20261031/audit.py';text=src.read_text()
    edits=[("ROOT / 'humanml_mirror_time_witness_v2_20261030/000008_joint_witness.npz'",
            "ROOT / 'accad_013267_fullrate_20261031/human_120hz.npz'"),
           ('len(rows) == report[\'window_count\'] == 10','len(rows) == report[\'window_count\'] == 88'),
           ('covered == set(range(1, 43))','covered == set(range(1, 355))')]
    for a,b in edits:
        if text.count(a)!=1:raise ValueError('unexpected template: '+a)
        text=text.replace(a,b)
    with (HERE/'audit.py').open('x',encoding='utf8',newline='\n') as f:f.write(text)
    with (HERE/'audit_manifest.json').open('x',encoding='utf8') as f:
        json.dump({'old_audit_sha256':sha(src),'new_audit_sha256':sha(HERE/'audit.py'),
                   'changes':'source path, 88 windows, source coverage 1..354','training_ready':False},f,indent=2);f.write('\n')


if __name__=='__main__':main()
