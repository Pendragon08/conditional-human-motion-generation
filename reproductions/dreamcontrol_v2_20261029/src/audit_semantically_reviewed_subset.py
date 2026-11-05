"""Check retained caption receipts, train-only statistics, grouping and closed gate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PRE=ROOT/'four_caption_preflight_20261101'
DATA=ROOT/'six_family_disjoint_preview_v2_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    report=json.loads((HERE/'report.json').read_text())
    assert sha(PRE/'manifest.json')==report['parent_manifest_sha256'] and sha(HERE/'manifest.json')==report['manifest_sha256']
    assert sha(HERE/'state_manifest.json')==report['state_manifest_sha256'] and sha(HERE/'stats.npz')==report['stats_sha256']
    old=json.loads((PRE/'manifest.json').read_text())['records'];now=json.loads((HERE/'manifest.json').read_text())['records']
    assert now==[r for r in old if r['caption_accepted'] is True] and set(report['excluded'])=={'000069','003017'}
    states={r['sample_id']:r for r in json.loads((HERE/'state_manifest.json').read_text())['records']}
    assert set(states)=={r['sample_id'] for r in now}
    groups={};train=[];spatial=[]
    for row in now:
        state=states[row['sample_id']];family=row['provenance_keys']['source_family'][0]
        assert groups.setdefault(family,row['split'])==row['split']
        assert sha(DATA/row['feature_path'])==row['feature_sha256']==state['feature_sha256']
        assert sha(DATA/state['state_path'])==state['state_sha256']
        if row['split']=='train':
            with np.load(DATA/row['feature_path'],allow_pickle=False) as f:x=f['features'].copy()
            train.append(x.astype(np.float64))
            with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64))
    a=np.concatenate(train);b=np.concatenate(spatial);assert len(a)==report['train_frames']==140
    expected={'feature_mean':a.mean(0),'feature_std':np.maximum(a.std(0),.001),
              'spatial_mean':b.mean(0),'spatial_std':np.maximum(b.std(0),.001)}
    errors={}
    with np.load(HERE/'stats.npz',allow_pickle=False) as f:
        for k,v in expected.items():
            errors[k]=float(np.abs(f[k].reshape(-1)-v).max());np.testing.assert_allclose(f[k].reshape(-1),v,rtol=1e-5,atol=1e-6)
    pre=json.loads(next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{')))
    assert pre['caption_evidence_audit']['verified_semantic_receipts']==4 and pre['caption_evidence_audit']['unaccepted_captions']==0
    assert pre['feature_audit']['training_ready'] is False and pre['state_audit']['state_technical_checks_passed']
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert not list(HERE.glob('*.pt'))
    out={'manifest_sha256':sha(HERE/'manifest.json'),'stats_sha256':sha(HERE/'stats.npz'),
         'counts':report['counts'],'train_frames':len(a),'caption_receipts':4,'max_stats_errors':errors,
         'excluded':report['excluded'],'entry_rejected_training':True,'training_ready':False,
         'script_sha256':sha(Path(__file__)),
         'limitations':'Same decoder; visual reviews, source identity, timing and physical quality remain separate from overall acceptance.'}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
