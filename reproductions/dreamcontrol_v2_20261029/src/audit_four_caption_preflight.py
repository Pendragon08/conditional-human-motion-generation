"""Verify the scoped MoSh review update and that training remains blocked."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'three_caption_preflight_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report=json.loads((HERE/'prepare_report.json').read_text())
    assert sha(BASE/'manifest.json')==report['parent_manifest_sha256'] and sha(HERE/'manifest.json')==report['manifest_sha256']
    assert sha(HERE/'caption_evidence_manifest.json')==report['caption_evidence_manifest_sha256']
    old=json.loads((BASE/'manifest.json').read_text())['records'];new=json.loads((HERE/'manifest.json').read_text())['records']
    assert len(old)==len(new)==6
    for a,b in zip(old,new):
        for key in ('sample_id','source_id','group_id','provenance_keys','split','feature_path','feature_sha256','fps'):
            assert a[key]==b[key]
        if a['sample_id']!='MOSH_000798_crop':assert a==b
        else:assert b['caption_accepted'] is True and b['caption_review_sha256']==report['new_review_sha256']
    receipts=json.loads((HERE/'caption_evidence_manifest.json').read_text())['receipts']
    assert len(receipts)==4 and len({x['sample_id'] for x in receipts})==4
    for e in receipts:
        review=ROOT/e['receipt_path'];audit=ROOT/e['independent_audit_path']
        assert sha(review)==e['receipt_sha256'] and sha(audit)==e['independent_audit_sha256']
        assert json.loads(audit.read_text())['receipt_sha256']==e['receipt_sha256']
        motion=next(x for x in new if x['sample_id']==e['sample_id']);r=json.loads(review.read_text())
        assert motion['caption']==r['caption'] and motion['feature_sha256']==r['candidate_sha256']
    pre=json.loads(next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{')))
    assert pre['caption_evidence_audit']['verified_semantic_receipts']==4 and pre['caption_evidence_audit']['unaccepted_captions']==2
    assert pre['feature_audit']['training_ready'] is False and pre['state_audit']['state_technical_checks_passed']
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert not list(HERE.glob('*.pt'))
    out={'manifest_sha256':sha(HERE/'manifest.json'),'caption_evidence_manifest_sha256':sha(HERE/'caption_evidence_manifest.json'),
         'verified_caption_receipts':4,'unaccepted_captions':2,'entry_rejected_training':True,'training_ready':False,
         'script_sha256':sha(Path(__file__)),
         'limitations':'Binding check only; no independent semantic reviewer or source, physical or training acceptance.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
