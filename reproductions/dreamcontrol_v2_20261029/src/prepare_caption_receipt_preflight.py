"""Bind a reviewed caption to an isolated existing preview configuration."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
REVIEW=ROOT/'ssm_011120_semantic_review_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def main():
    audit=json.loads((BASE/'independent_audit.json').read_text())
    assert sha(BASE/'manifest.json')==audit['manifest_sha256'] and sha(BASE/'stats.npz')==audit['stats_sha256']
    rev=REVIEW/'semantic_review.json';check=REVIEW/'independent_audit.json'
    assert json.loads(check.read_text())['receipt_sha256']==sha(rev)
    records=json.loads((BASE/'manifest.json').read_text())['records']
    assert [r['sample_id'] for r in records if r.get('caption_accepted') is True]==['011120']
    bundle={'receipts':[{'sample_id':'011120','receipt_path':str(rev.relative_to(ROOT)),'receipt_sha256':sha(rev),
                         'independent_audit_path':str(check.relative_to(ROOT)),'independent_audit_sha256':sha(check)}]}
    save(HERE/'caption_evidence_manifest.json',bundle)
    cfg=json.loads((BASE/'preflight_config.json').read_text())
    cfg.update({'caption_evidence_root':str(ROOT),'caption_evidence_manifest':str(HERE/'caption_evidence_manifest.json'),
                'run_dir':str(HERE),'purpose':'isolated fail-closed caption receipt preflight'})
    save(HERE/'config.json',cfg)
    save(HERE/'prepare_report.json',{'parent_manifest_sha256':sha(BASE/'manifest.json'),
          'caption_evidence_manifest_sha256':sha(HERE/'caption_evidence_manifest.json'),
          'config_sha256':sha(HERE/'config.json'),'training_ready':False})


if __name__=='__main__':main()
