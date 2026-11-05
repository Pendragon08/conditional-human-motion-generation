"""Independent bookkeeping and train-only statistics audit; not acceptance."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');HERE=Path(__file__).resolve().parent
BASE=ROOT/'six_punch_manual_preview_20261101';DATA=ROOT/'ssm_new6_20hz_resume_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    build=json.loads((HERE/'build_report.json').read_text());records=json.loads((HERE/'manifest.json').read_text())['records']
    states=json.loads((HERE/'state_manifest.json').read_text())['records'];by={r['sample_id']:r for r in states}
    pre=next(json.loads(x) for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{'))
    assert sha(BASE/'manifest.json')==build['base_manifest_sha256'] and sha(DATA/'report.json')==build['new_data_report_sha256']
    assert sha(HERE/'manifest.json')==build['manifest_sha256'] and sha(HERE/'state_manifest.json')==build['state_manifest_sha256'] and sha(HERE/'stats.npz')==build['stats_sha256']
    assert records[:6]==json.loads((BASE/'manifest.json').read_text())['records'] and len(records)==len(by)==8
    new={r['sample_id']:r for r in records[6:]};assert set(new)=={'007129','004028'}
    assert new['007129']['caption_origin']=='source_text' and new['004028']['caption_origin']=='manual_visual'
    assert new['007129']['caption_review_sha256']==sha(ROOT/'ssm_007129_caption_review_20261101/007129_review.json')
    assert new['004028']['caption_review_sha256']==sha(ROOT/'ssm_004028_caption_review_20261101/004028_annotation.json')
    arrays=[];spatial=[];keys={}
    for r in records:
        state=by[r['sample_id']];feature=HERE/r['feature_path'];p=HERE/state['state_path']
        assert sha(feature)==r['feature_sha256'] and sha(p)==state['state_sha256'] and state['feature_sha256']==r['feature_sha256']
        for namespace,values in r['provenance_keys'].items():
            for value in values:
                k=(namespace,value);assert keys.setdefault(k,r['split'])==r['split']
        with np.load(feature,allow_pickle=False) as f:x=f['features'].copy()
        if r['split']=='train':
            arrays.append(x.astype(np.float64))
            with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64))
    x=np.concatenate(arrays);p=np.concatenate(spatial)
    with np.load(HERE/'stats.npz',allow_pickle=False) as f:
        errors=[float(np.max(abs(f['feature_mean']-x.mean(0)))),float(np.max(abs(f['feature_std']-np.maximum(x.std(0),.001)))),
                float(np.max(abs(f['spatial_mean']-p.mean(0)))),float(np.max(abs(f['spatial_std']-np.maximum(p.std(0),.001))))]
    assert len(x)==build['train_feature_frames']==437 and max(errors)<1e-6
    assert pre['feature_audit']['counts']=={'train':6,'val':1,'test':1} and pre['state_audit']['state_technical_checks_passed']
    assert pre['caption_evidence_audit']['verified_semantic_receipts']==5
    assert len(pre['caption_archive_binding']['cases'])==5 and len(pre['manual_caption_binding']['cases'])==3
    assert pre['ssm_source_evidence_audit']['marker_evidence_cases']==5 and pre['feature_audit']['training_ready'] is False
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert not list(HERE.glob('*.pt'))
    result={'manifest_sha256':sha(HERE/'manifest.json'),'stats_sha256':sha(HERE/'stats.npz'),
        'preflight_log_sha256':sha(HERE/'preflight.log'),'script_sha256':sha(Path(__file__)),
        'split_counts':pre['feature_audit']['counts'],'train_feature_frames':len(x),'stats_max_abs_differences':errors,
        'source_caption_receipts':5,'manual_caption_receipts':3,'ssm_marker_cases':5,
        'training_rejected':True,'training_ready':False,
        'limitations':['Spatial stats and movement encoder share existing decoder.',
          'Caption reviews are bounded subjective judgments without independent semantic reviewer.',
          'Same SSM recording family; actual clock, source/license and physics remain unaccepted.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
