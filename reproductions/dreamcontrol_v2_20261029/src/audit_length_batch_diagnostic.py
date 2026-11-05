"""Recheck actual lengths, baseline source lines, and sampler coverage."""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cfg=json.loads((HERE/'config.json').read_text());settings=cfg['legacy_length_diagnostic']
    source=Path(settings['legacy_dataset']);opts=Path(settings['legacy_options'])
    assert sha(source)==settings['legacy_dataset_sha256'] and sha(opts)==settings['legacy_options_sha256']
    text=source.read_text();opt=opts.read_text()
    assert "min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24" in text
    assert '(len(motion)) < min_motion_len or (len(motion) >= 200)' in text
    assert 'dataset_name: t2m' in opt
    manifest=json.loads(Path(cfg['manifest']).read_text())['records'];root=Path(cfg['dataset_root'])
    pre=json.loads(next(x for x in (HERE/'preflight.log').read_text().splitlines() if x.startswith('{')))
    diag=pre['legacy_length_and_batch_diagnostic'];by_id={r['sample_id']:r for r in diag['cases']}
    train=[];lengths={}
    for rec in manifest:
        path=root/rec['feature_path'];assert sha(path)==rec['feature_sha256']
        with np.load(path,allow_pickle=False) as f:n=len(f['features'])
        out=by_id[rec['sample_id']];assert out['feature_rows']==n
        assert out['legacy_t2m_whole_motion_length_eligible']==(40<=n<200)
        assert out['experimental_loader_length_eligible']==(2<=n<=199)
        if rec['split']=='train':train.append(rec['sample_id']);lengths[rec['sample_id']]=n
    batches=diag['actual_train_batches'];flatten=[sid for batch in batches for sid in batch]
    assert collections.Counter(flatten)==collections.Counter(train)
    for batch in batches:assert 1<=len(batch)<=cfg['batch_size'] and len({lengths[x] for x in batch})==1
    histogram=dict(collections.Counter(str(len(x)) for x in batches));assert histogram==diag['actual_batch_size_histogram']
    assert 'Ran 2 tests' in (HERE/'tests.log').read_text() and 'OK' in (HERE/'tests.log').read_text()
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    out={'config_sha256':sha(HERE/'config.json'),'preflight_log_sha256':sha(HERE/'preflight.log'),
         'script_sha256':sha(Path(__file__)),'cases':diag['cases'],'actual_train_batches':batches,
         'configured_batch_size':cfg['batch_size'],'actual_batch_size_histogram':histogram,
         'training_ready':False,'limitations':['Legacy code policy is not a verified v2 dataset filter.',
         'No automatic exclusion of GRAB; batch sizes are for this exact preview/configuration only.']}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
