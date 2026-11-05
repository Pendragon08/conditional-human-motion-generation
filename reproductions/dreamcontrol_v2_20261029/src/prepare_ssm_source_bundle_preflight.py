"""Bind the two validated SSM source receipts to a closed training preflight."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PRE=ROOT/'four_semantic_subset_20261101'
SOURCE=ROOT/'ssm_source_preflight_v2_20261101'
ALIGN=ROOT/'ssm_id_alignment_20261030/ids.jsonl'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def main():
    audited=json.loads((PRE/'audit.json').read_text());source=json.loads((SOURCE/'report.json').read_text())
    assert sha(PRE/'manifest.json')==audited['manifest_sha256']==source['manifest_sha256']
    assert sha(ALIGN)==source['alignment_sha256']
    entries=[{'sample_id':x['sample_id'],'receipt_sha256':x['receipt_sha256']} for x in source['cases']]
    assert {x['sample_id'] for x in entries}=={'000008','011120'}
    save(HERE/'source_evidence_manifest.json',{'alignment_sha256':sha(ALIGN),'receipts':entries})
    cfg=json.loads((PRE/'config.json').read_text());cfg.update({'ssm_alignment_path':str(ALIGN),
       'ssm_source_evidence_manifest':str(HERE/'source_evidence_manifest.json'),'run_dir':str(HERE),
       'purpose':'closed source-marker preflight; source and physical acceptance incomplete'})
    save(HERE/'config.json',cfg)
    save(HERE/'prepare.json',{'source_report_sha256':sha(SOURCE/'report.json'),'source_evidence_manifest_sha256':sha(HERE/'source_evidence_manifest.json'),
         'config_sha256':sha(HERE/'config.json'),'training_ready':False})


if __name__=='__main__':main()
