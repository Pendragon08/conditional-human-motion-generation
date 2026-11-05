"""Independently recount a diagnostic preview, its statistics and fail-closed entry."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'five_conservative_preview_20261031'
SEM=ROOT/'ssm_011120_semantic_review_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    report=json.loads((HERE/'report.json').read_text())
    assert sha(HERE/'manifest.json')==report['manifest_sha256'] and sha(HERE/'state_manifest.json')==report['state_manifest_sha256']
    assert sha(HERE/'stats.npz')==report['stats_sha256'] and sha(BASE/'manifest.json')==report['parent_manifest_sha256']
    assert sha(SEM/'semantic_review.json')==report['semantic_receipt_sha256']
    records=json.loads((HERE/'manifest.json').read_text())['records']; prior=json.loads((BASE/'manifest.json').read_text())['records']
    states=json.loads((HERE/'state_manifest.json').read_text())['records']
    assert records[:5]==prior and len(records)==len(states)==6
    state_by_id={r['sample_id']:r for r in states}; train=[]; pts=[]; split_groups={}; keys={}
    for rec in records:
        s=state_by_id[rec['sample_id']]
        assert sha(HERE/rec['feature_path'])==rec['feature_sha256']==s['feature_sha256']
        assert sha(HERE/s['state_path'])==s['state_sha256']
        assert split_groups.setdefault(rec['group_id'],rec['split'])==rec['split']
        for k,values in rec['provenance_keys'].items():
            for v in values:assert keys.setdefault((k,v),rec['split'])==rec['split']
        if rec['split']=='train':
            with np.load(HERE/rec['feature_path'],allow_pickle=False) as d:x=d['features'].copy()
            train.append(x.astype(np.float64))
            with torch.no_grad():pts.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64))
    added=records[-1]; receipt=json.loads((SEM/'semantic_review.json').read_text())
    assert added['sample_id']=='011120' and added['caption']==receipt['caption'] and added['caption_accepted'] is True
    assert added['caption_review_sha256']==report['semantic_receipt_sha256'] and added['feature_sha256']==receipt['candidate_sha256']
    assert all(r.get('caption_accepted') is False for r in records[:5])
    all_train=np.concatenate(train); all_points=np.concatenate(pts)
    assert len(all_train)==report['train_frames']==294
    expected={'feature_mean':all_train.mean(0),'feature_std':np.maximum(all_train.std(0),.001),
              'spatial_mean':all_points.mean(0),'spatial_std':np.maximum(all_points.std(0),.001)}
    errors={}
    with np.load(HERE/'stats.npz',allow_pickle=False) as data:
        for name,want in expected.items():
            errors[name]=float(np.max(np.abs(data[name].reshape(-1)-want)))
            np.testing.assert_allclose(data[name].reshape(-1),want,rtol=1e-5,atol=1e-6)
    raw=(HERE/'preflight.log').read_text(); line=next(x for x in raw.splitlines() if x.startswith('{'))
    pre=json.loads(line)
    assert pre['feature_audit']['technical_checks_passed'] and pre['state_audit']['state_technical_checks_passed']
    assert pre['feature_audit']['caption_status_counts']=={'declared_accepted':1,'unaccepted_or_unspecified':5}
    assert pre['feature_audit']['training_ready'] is False
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert not list(HERE.glob('*.pt'))
    out={'manifest_sha256':sha(HERE/'manifest.json'),'stats_sha256':sha(HERE/'stats.npz'),
         'train_frames':len(all_train),'counts':{k:sum(r['split']==k for r in records) for k in ('train','val','test')},
         'statistics_max_abs_errors':errors,'caption_reviewed':1,'captions_unaccepted':5,
         'entry_rejected_training':True,'training_ready':False,'script_sha256':sha(Path(__file__)),
         'limitations':'Shares decoder; does not independently establish semantic truth, physical quality or source identity.'}
    with (HERE/'independent_audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
