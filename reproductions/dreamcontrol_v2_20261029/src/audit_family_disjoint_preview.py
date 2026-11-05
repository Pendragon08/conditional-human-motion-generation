"""Recompute source-family split, all training statistics and gate rejection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_reviewed_caption_preview_20261101'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    rep=json.loads((HERE/'report.json').read_text()); old=json.loads((BASE/'manifest.json').read_text())['records']
    records=json.loads((HERE/'manifest.json').read_text())['records']; states=json.loads((HERE/'state_manifest.json').read_text())['records']
    assert sha(BASE/'manifest.json')==rep['parent_manifest_sha256'] and sha(HERE/'manifest.json')==rep['manifest_sha256']
    assert sha(HERE/'state_manifest.json')==rep['state_manifest_sha256'] and sha(HERE/'stats.npz')==rep['stats_sha256']
    assert len(old)==len(records)==len(states)==6
    states={r['sample_id']:r for r in states}; groups={}; parents={r['sample_id']:r for r in old}; train=[]; spatial=[]
    for row in records:
        before=parents[row['sample_id']]; state=states[row['sample_id']]
        for k in ('sample_id','source_id','caption','caption_accepted','feature_sha256','feature_path','fps'):
            assert row[k]==before[k]
        path=row['source_id'].removeprefix('./'); parts=path.split('/')
        assert parts[0]=='pose_data' and parts[1] in ('SSM_synced','ACCAD','MPI_mosh')
        family=parts[1]; expected={'SSM_synced':'train','ACCAD':'val','MPI_mosh':'test'}[family]
        assert row['split']==expected and row['provenance_keys']['source_family']==[family]
        assert groups.setdefault(family,row['split'])==row['split']
        assert sha(HERE/row['feature_path'])==row['feature_sha256']==state['feature_sha256']
        assert sha(HERE/state['state_path'])==state['state_sha256']
        if row['split']=='train':
            with np.load(HERE/row['feature_path'],allow_pickle=False) as f:x=f['features'].copy()
            train.append(x.astype(np.float64))
            with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64))
    features=np.concatenate(train); points=np.concatenate(spatial)
    expect={'feature_mean':features.mean(0),'feature_std':np.maximum(features.std(0),.001),
            'spatial_mean':points.mean(0),'spatial_std':np.maximum(points.std(0),.001)}
    errors={}
    with np.load(HERE/'stats.npz',allow_pickle=False) as f:
        for k,v in expect.items():
            errors[k]=float(np.abs(f[k].reshape(-1)-v).max());np.testing.assert_allclose(f[k].reshape(-1),v,rtol=1e-5,atol=1e-6)
    assert len(features)==rep['train_frames']==231 and rep['counts']=={'train':4,'val':1,'test':1}
    pre=json.loads(next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{')))
    assert pre['feature_audit']['technical_checks_passed'] and pre['state_audit']['state_technical_checks_passed']
    assert pre['feature_audit']['training_ready'] is False
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    assert not list(HERE.glob('*.pt'))
    result={'manifest_sha256':sha(HERE/'manifest.json'),'stats_sha256':sha(HERE/'stats.npz'),
            'family_splits':groups,'counts':rep['counts'],'train_frames':len(features),'max_stats_errors':errors,
            'entry_rejected_training':True,'training_ready':False,'script_sha256':sha(Path(__file__)),
            'limitations':'Path families are not verified subject identities; spatial statistics share existing decoder; no physical/semantic acceptance.'}
    with (HERE/'independent_audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
