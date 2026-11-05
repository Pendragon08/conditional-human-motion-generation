"""Reuse the audited full-rate 013267 window/feature pipeline for 004347."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def adapt(src,dst):
    text=src.read_text(encoding='utf8')
    if '013267' not in text:raise ValueError('unexpected template: '+str(src))
    with dst.open('x',encoding='utf8',newline='\n') as f:f.write(text.replace('013267','004347'))
    return {'source':str(src),'source_sha256':sha(src),'new_sha256':sha(dst)}


def main():
    source=ROOT/'accad_013267_pyroki_fullrate_20261031';target=ROOT/'accad_004347_pyroki_fullrate_20261031'
    target.mkdir()
    out=[adapt(source/'prepare.py',target/'prepare.py'),adapt(source/'prepare_audit.py',target/'prepare_audit.py'),
         adapt(source/'queue.sh',target/'queue.sh')]
    info={'adaptations':out,'change':'013267 -> 004347','training_ready':False}
    with (target/'template_manifest.json').open('x',encoding='utf8') as f:json.dump(info,f,indent=2);f.write('\n')
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
