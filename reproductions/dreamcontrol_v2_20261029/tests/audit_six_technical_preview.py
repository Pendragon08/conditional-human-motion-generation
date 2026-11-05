"""Independently recalculate a non-training candidate preview's statistics."""
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    report=json.loads((HERE/'report.json').read_text())
    manifestpath=HERE/'experimental_manifest_not_for_training.json'
    statspath=HERE/'experimental_stats_not_for_training.npz'
    assert sha(manifestpath)==report['manifest_sha256'] and sha(statspath)==report['stats_sha256']
    manifest=json.loads(manifestpath.read_text()); assert manifest['training_ready'] is False
    old=ROOT/'ssm_four_preview_20261031'
    previous={r['sample_id']:r for r in json.loads((old/'experimental_manifest_not_for_training.json').read_text())['records']}
    view=ROOT/'source_grouping_v3_20261031/six_source_preview.json'
    assert sha(view)==report['view_sha256']
    source_view={r['sample_id']:r for r in json.loads(view.read_text())['records']}
    split_by_key={}; counts=Counter(); train=[]; train_points=[]; caption_checks=0
    with zipfile.ZipFile(HUMANML/'texts.zip') as z:
        for rec in manifest['records']:
            sample=rec['sample_id']; ref=source_view[sample]
            assert rec['training_ready'] is False and rec['caption_accepted'] is False
            assert rec['source_id']==ref['source_id'] and rec['split']==ref['split']
            assert rec['provenance_keys']==ref['provenance_keys']
            for namespace, values in rec['provenance_keys'].items():
                for value in values:
                    key=(namespace,value); prev=split_by_key.setdefault(key,rec['split'])
                    assert prev==rec['split']
            path=HERE/rec['feature_path'];assert sha(path)==rec['feature_sha256']
            if sample in previous:
                prior=previous[sample];assert sha(old/prior['feature_path'])==rec['feature_sha256']
                assert rec['caption']==prior['caption'] and rec['caption_source']==prior['caption_source']
            else:
                label=sample[4:]; bml=ROOT/'bml_joint_smoothness_features_20261031'/label
                checked=json.loads((bml/'audit_report.json').read_text())
                assert sha(bml/'candidate.npz')==rec['feature_sha256']==checked['feature_sha256']
                ann=ref['candidate_clip_annotations'][0]
                lines=z.read('texts/'+ann['clip_id']+'.txt').decode('utf8').splitlines()
                assert rec['caption']==lines[0].split('#')[0] and rec['caption']==ann['annotations'][0]['caption']
                assert rec['caption_source']=='HumanML texts.zip: texts/'+ann['clip_id']+'.txt'
            caption_checks+=1;counts[rec['split']]+=1
            with np.load(path,allow_pickle=False) as f:x=f['features'].copy()
            if rec['split']=='train':
                train.append(x.astype(np.float64))
                with torch.no_grad():p=recover_from_ric(torch.from_numpy(x),22).numpy()
                train_points.append(p.reshape(len(p),66).astype(np.float64))
    data=np.concatenate(train); points=np.concatenate(train_points)
    with np.load(statspath,allow_pickle=False) as f:
        expected={'feature_mean':data.mean(0),'feature_std':np.maximum(data.std(0),.001),
                  'spatial_mean':points.mean(0),'spatial_std':np.maximum(points.std(0),.001)}
        errors={k:float(np.max(np.abs(f[k]-v))) for k,v in expected.items()}
    assert max(errors.values())<1e-6 and len(data)==report['train_frames']
    assert counts==Counter({'train':2,'val':1,'test':3}) and caption_checks==6
    out={'status':'passed','manifest_sha256':sha(manifestpath),'stats_sha256':sha(statspath),
         'counts':dict(counts),'train_frames':len(data),'statistics_max_abs_errors':errors,
         'captions_traced':caption_checks,'training_ready':False,
         'limitations':['Caption text provenance checked, not semantic correctness.',
                        'Uses same joint decoder; no independent FK, encoding, physical or model verification.',
                        'Grouping depends on declared provenance keys.']}
    with (HERE/'independent_audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
