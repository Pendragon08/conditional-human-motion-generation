"""Create isolated scripts for a second complete, short ACCAD candidate.

The tested 013267 pipeline is reused through a single literal ID substitution;
outputs, inputs and logs are never overwritten. Result remains unaccepted.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_adapted(src,dst,*,require_id=True):
    text=src.read_text(encoding='utf8')
    if require_id and '013267' not in text:raise ValueError('unexpected template '+str(src))
    out=text.replace('013267','004347')
    with dst.open('x',encoding='utf8',newline='\n') as f:f.write(out)
    return {'source':str(src),'source_sha256':sha(src),'output_sha256':sha(dst),
            'change':'013267 -> 004347'}


def main():
    human=ROOT/'accad_004347_fullrate_20261031';human.mkdir()
    pairs=[(ROOT/'accad_013267_fullrate_20261031/reconstruct_v2.py',human/'reconstruct.py'),
           (ROOT/'accad_013267_fullrate_20261031/audit_human.py',human/'audit_human_template.py')]
    details=[write_adapted(a,b,require_id=(i==0)) for i,(a,b) in enumerate(pairs)]
    # The original human auditor pins the filename reconstruct_v2.py.
    p=human/'audit_human_template.py';text=p.read_text()
    old="HERE/'reconstruct_v2.py'";new="HERE/'reconstruct.py'"
    if text.count(old)!=1:raise ValueError('unexpected auditor filename')
    with (human/'audit_human.py').open('x',encoding='utf8',newline='\n') as f:f.write(text.replace(old,new))
    with (human/'prepare_manifest.json').open('x',encoding='utf8') as f:
        json.dump({'adaptations':details,'audit_sha256':sha(human/'audit_human.py'),
                   'training_ready':False},f,indent=2);f.write('\n')
    print(json.dumps({'human_dir':str(human),'preparation_sha256':sha(human/'prepare_manifest.json')}),flush=True)


if __name__=='__main__':main()
