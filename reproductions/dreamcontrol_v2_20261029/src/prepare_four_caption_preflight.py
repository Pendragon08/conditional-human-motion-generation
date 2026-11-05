"""Bind one additional crop-specific visual review to a diagnostic manifest."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'three_caption_preflight_20261101'
NEW=ROOT/'mosh_caption_semantic_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def main():
    prior=json.loads((BASE/'audit.json').read_text())
    assert sha(BASE/'manifest.json')==prior['manifest_sha256']
    assert sha(BASE/'caption_evidence_manifest.json')==prior['caption_evidence_manifest_sha256']
    manifest=json.loads((BASE/'manifest.json').read_text());bundle=json.loads((BASE/'caption_evidence_manifest.json').read_text())
    r=next(x for x in manifest['records'] if x['sample_id']=='MOSH_000798_crop')
    receipt=NEW/'semantic_review.json';audit=NEW/'independent_audit.json';review=json.loads(receipt.read_text())
    assert json.loads(audit.read_text())['receipt_sha256']==sha(receipt)
    assert review['sample_id']==r['sample_id'] and review['candidate_sha256']==r['feature_sha256']
    assert review['manual_semantic_assessment']['semantic_accepted_for_exported_samples'] is True
    r['caption']=review['caption'];r['caption_accepted']=True;r['caption_review_sha256']=sha(receipt)
    bundle['receipts'].append({'sample_id':r['sample_id'],'receipt_path':str(receipt.relative_to(ROOT)),
       'receipt_sha256':sha(receipt),'independent_audit_path':str(audit.relative_to(ROOT)),'independent_audit_sha256':sha(audit)})
    assert len(bundle['receipts'])==4 and len({x['sample_id'] for x in bundle['receipts']})==4
    save(HERE/'manifest.json',manifest);save(HERE/'caption_evidence_manifest.json',bundle)
    cfg=json.loads((BASE/'config.json').read_text());cfg.update({'manifest':str(HERE/'manifest.json'),
        'caption_evidence_manifest':str(HERE/'caption_evidence_manifest.json'),'run_dir':str(HERE),
        'purpose':'isolated four-caption review preflight; dataset unaccepted'})
    save(HERE/'config.json',cfg)
    save(HERE/'prepare_report.json',{'parent_manifest_sha256':sha(BASE/'manifest.json'),
        'manifest_sha256':sha(HERE/'manifest.json'),'caption_evidence_manifest_sha256':sha(HERE/'caption_evidence_manifest.json'),
        'new_review_sha256':sha(receipt),'training_ready':False})


if __name__=='__main__':main()
