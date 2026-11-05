"""Prepare an unaccepted preview with three source-bound semantic reviews."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
FIRST=ROOT/'ssm_011120_semantic_review_20261101'
NEW=ROOT/'two_caption_reviews_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def main():
    parent=json.loads((BASE/'independent_audit.json').read_text())
    assert sha(BASE/'manifest.json')==parent['manifest_sha256'] and sha(BASE/'stats.npz')==parent['stats_sha256']
    manifest=json.loads((BASE/'manifest.json').read_text()); newaudit=json.loads((NEW/'independent_audit.json').read_text())
    assertions={x['sample_id']:x for x in newaudit['cases']}; bundle=[]
    for rec in manifest['records']:
        sid=rec['sample_id']
        if sid=='011120':receipt=FIRST/'semantic_review.json';audit=FIRST/'independent_audit.json'
        elif sid in ('000008','ACCAD_004347'):
            receipt=NEW/(sid+'_review.json');audit=NEW/(sid+'_audit.json')
            assert sha(receipt)==assertions[sid]['receipt_sha256'] and sha(audit)==assertions[sid]['audit_sha256']
        else:continue
        review=json.loads(receipt.read_text());checked=json.loads(audit.read_text())
        assert checked['receipt_sha256']==sha(receipt) and review['candidate_sha256']==rec['feature_sha256']
        assert review['manual_semantic_assessment']['semantic_accepted_for_exported_samples'] is True
        rec['caption']=review['caption'];rec['caption_accepted']=True;rec['caption_review_sha256']=sha(receipt)
        bundle.append({'sample_id':sid,'receipt_path':str(receipt.relative_to(ROOT)),'receipt_sha256':sha(receipt),
                       'independent_audit_path':str(audit.relative_to(ROOT)),'independent_audit_sha256':sha(audit)})
    assert {x['sample_id'] for x in bundle}=={'011120','000008','ACCAD_004347'}
    manifest['training_ready']=False
    save(HERE/'manifest.json',manifest);save(HERE/'caption_evidence_manifest.json',{'receipts':bundle})
    cfg=json.loads((BASE/'preflight_config.json').read_text())
    cfg.update({'manifest':str(HERE/'manifest.json'),'caption_evidence_root':str(ROOT),
                'caption_evidence_manifest':str(HERE/'caption_evidence_manifest.json'),
                'run_dir':str(HERE),'purpose':'isolated three-caption review preflight; unaccepted dataset'})
    save(HERE/'config.json',cfg)
    save(HERE/'prepare_report.json',{'parent_manifest_sha256':sha(BASE/'manifest.json'),
        'manifest_sha256':sha(HERE/'manifest.json'),'caption_evidence_manifest_sha256':sha(HERE/'caption_evidence_manifest.json'),
        'new_review_audit_sha256':sha(NEW/'independent_audit.json'),'training_ready':False})


if __name__=='__main__':main()
