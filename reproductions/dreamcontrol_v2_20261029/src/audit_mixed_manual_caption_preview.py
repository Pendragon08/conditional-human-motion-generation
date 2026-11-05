"""Independently verify the mixed preview; it remains unaccepted."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'four_semantic_subset_20261101';GRAB=ROOT/'grab_s3_annotated_preview_20261101';HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    build=json.loads((HERE/'build_report.json').read_text());manifest=json.loads((HERE/'manifest.json').read_text())
    states=json.loads((HERE/'state_manifest.json').read_text())['records']
    pre=json.loads(next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{')))
    assert sha(BASE/'manifest.json')==build['parent_manifest_sha256'] and sha(GRAB/'manifest.json')==build['grab_manifest_sha256']
    assert sha(HERE/'manifest.json')==build['manifest_sha256'] and sha(HERE/'state_manifest.json')==build['state_manifest_sha256']
    assert sha(HERE/'stats.npz')==build['stats_sha256'] and len(manifest['records'])==5
    parent=json.loads((BASE/'manifest.json').read_text())['records'];assert manifest['records'][:4]==parent
    manual=manifest['records'][-1];assert manual['caption_origin']=='manual_visual' and manual['caption']=='a person lowers their right arm.'
    assert manual['split']=='train' and manual['training_ready'] is False and manifest['training_ready'] is False
    state_idx={r['sample_id']:r for r in states};feature=[];spatial=[];groups={};tot=0
    for row in manifest['records']:
        p=HERE/row['feature_path'];s=state_idx[row['sample_id']]
        assert sha(p)==row['feature_sha256'] and sha(HERE/s['state_path'])==s['state_sha256']
        old=groups.setdefault(row['group_id'],row['split']);assert old==row['split']
        with np.load(p,allow_pickle=False) as f:x=f['features'].copy()
        if row['split']=='train':
            feature.append(x.astype(np.float64));tot+=len(x)
            with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64))
    x=np.concatenate(feature);p=np.concatenate(spatial)
    with np.load(HERE/'stats.npz',allow_pickle=False) as s:
        errors=[float(np.max(np.abs(s['feature_mean']-x.mean(0)))),
                float(np.max(np.abs(s['feature_std']-np.maximum(x.std(0),.001)))),
                float(np.max(np.abs(s['spatial_mean']-p.mean(0)))),
                float(np.max(np.abs(s['spatial_std']-np.maximum(p.std(0),.001))))]
    assert tot==159==build['train_feature_frames'] and max(errors)<1e-6
    assert pre['feature_audit']['counts']=={'train':3,'val':1,'test':1}
    assert pre['state_audit']['state_technical_checks_passed'] is True
    assert pre['caption_evidence_audit']['verified_semantic_receipts']==4
    assert len(pre['caption_archive_binding']['cases'])==4 and len(pre['manual_caption_binding']['cases'])==1
    assert pre['manual_caption_binding']['cases'][0]['sample_id']==manual['sample_id']
    assert len(pre['vertical_support_diagnostic']['cases'])==5 and pre['feature_audit']['training_ready'] is False
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert 'Ran 2 tests' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'tests.log').read_text()
    out={'build_report_sha256':sha(HERE/'build_report.json'),'manifest_sha256':sha(HERE/'manifest.json'),
         'preflight_log_sha256':sha(HERE/'preflight.log'),'script_sha256':sha(Path(__file__)),
         'split_counts':pre['feature_audit']['counts'],'train_frames':tot,'stats_max_abs_differences':errors,
         'source_caption_receipts':4,'manual_caption_receipts':1,'vertical_diagnostic':pre['vertical_support_diagnostic']['cases'],
         'run_rejected':True,'training_ready':False,
         'limitations':['Statistics independently recomputed but spatial points use the same decoder.',
                        'Caption and provenance bindings do not prove semantics, source clock, license or physics.',
                        'Tiny auxiliary dataset is not an adequate training or evaluation set.']}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
