"""Verify scoped caption changes and blocked training on the six-clip preview."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=json.loads((HERE/'prepare_report.json').read_text())
    assert sha(BASE/'manifest.json')==p['parent_manifest_sha256'] and sha(HERE/'manifest.json')==p['manifest_sha256']
    assert sha(HERE/'caption_evidence_manifest.json')==p['caption_evidence_manifest_sha256']
    old=json.loads((BASE/'manifest.json').read_text())['records'];new=json.loads((HERE/'manifest.json').read_text())['records']
    assert len(old)==len(new)==6
    for a,b in zip(old,new):
        for k in ('sample_id','source_id','group_id','provenance_keys','split','feature_path','feature_sha256','fps'):
            assert a[k]==b[k]
        if a['sample_id'] not in ('000008','ACCAD_004347','011120'):assert a==b
        else:assert b['caption_accepted'] is True
    bundle=json.loads((HERE/'caption_evidence_manifest.json').read_text())['receipts']
    assert {x['sample_id'] for x in bundle}=={'000008','ACCAD_004347','011120'}
    for r in bundle:
        receipt=ROOT/r['receipt_path'];audit=ROOT/r['independent_audit_path']
        assert sha(receipt)==r['receipt_sha256'] and sha(audit)==r['independent_audit_sha256']
        assert json.loads(audit.read_text())['receipt_sha256']==r['receipt_sha256']
        entry=next(x for x in new if x['sample_id']==r['sample_id']);content=json.loads(receipt.read_text())
        assert content['caption']==entry['caption'] and content['candidate_sha256']==entry['feature_sha256']
    pre=json.loads(next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{')))
    assert pre['caption_evidence_audit']['verified_semantic_receipts']==3 and pre['caption_evidence_audit']['unaccepted_captions']==3
    assert pre['feature_audit']['training_ready'] is False and pre['state_audit']['state_technical_checks_passed']
    cfg=json.loads((HERE/'config.json').read_text())
    assert sha(Path(cfg['stats']))==json.loads((BASE/'report.json').read_text())['stats_sha256']
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert not list(HERE.glob('*.pt'))
    out={'manifest_sha256':sha(HERE/'manifest.json'),'caption_evidence_manifest_sha256':sha(HERE/'caption_evidence_manifest.json'),
         'verified_caption_receipts':3,'unaccepted_captions':3,'actual_training_rejected':True,'training_ready':False,
         'script_sha256':sha(Path(__file__)),
         'limitations':'Verifies evidence binding, not independent semantic truth, source, physical or overall training acceptance.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
