"""Summarize real and negative preflight runs with pinned hashes."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    line=next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{'))
    data=json.loads(line); c=data['caption_evidence_audit']
    assert c['verified_semantic_receipts']==1 and c['unaccepted_captions']==5 and c['training_ready'] is False
    assert data['feature_audit']['training_ready'] is False and data['state_audit']['state_technical_checks_passed']
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert 'declared accepted captions require bound review receipts' in (HERE/'missing_evidence_rejected.log').read_text()
    assert 'OK' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'regression.log').read_text()
    assert not list(HERE.glob('*.pt'))
    result={'manifest_sha256':data['manifest_sha256'],'stats_sha256':data['stats_sha256'],
            'caption_evidence_manifest_sha256':c['evidence_manifest_sha256'],
            'verified_semantic_receipts':1,'unaccepted_captions':5,'missing_evidence_rejected':True,
            'actual_training_rejected':True,'training_ready':False,
            'files':{n:sha(HERE/n) for n in ('audit_caption_receipts.py','train_robot_prior.py','tests.log','regression.log',
                 'preflight.log','run_rejected.log','missing_evidence_rejected.log')},
            'limitations':'Structural caption evidence validation only; source, physical and overall acceptance not implemented.'}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
