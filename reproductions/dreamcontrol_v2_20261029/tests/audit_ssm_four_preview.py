"""Audit four-source non-training preview without changing the training gate."""
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT = Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HUMANML = Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D')
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    report = json.loads((HERE / 'report.json').read_text())
    manifest_file = HERE / 'experimental_manifest_not_for_training.json'
    stats_file = HERE / 'experimental_stats_not_for_training.npz'
    manifest = json.loads(manifest_file.read_text())
    assert report['training_ready'] is manifest['training_ready'] is False
    assert report['feature_audit']['technical_checks_passed'] is True
    assert report['feature_audit']['training_ready'] is False
    assert sha(HERE / 'build.py') == report['script_sha256']
    assert sha(manifest_file) == report['manifest_sha256']
    assert sha(stats_file) == report['stats_sha256']
    alignment = ROOT / 'ssm_id_alignment_20261030/ids.jsonl'
    assert sha(alignment) == report['source_mapping_sha256']
    mapping = {r['motion_id']: r for r in map(json.loads,alignment.read_text().splitlines())}
    original = {name:set((HUMANML / (name+'.txt')).read_text().splitlines()) for name in ('train','val','test')}
    archive = HUMANML / 'texts.zip'
    assert sha(archive) == report['caption_archive_sha256']
    samples = {s['id']: s for s in report['samples']}
    records = manifest['records']
    expected = {'000008':'train','000069':'train','013303':'val','003017':'test'}
    assert {r['sample_id']:r['split'] for r in records} == expected
    seen_groups, train_features, train_positions = {}, [], []
    for rec in records:
        name = rec['sample_id']; item = samples[name]; source = mapping[name]['source_path']
        assert rec['training_ready'] is False and not mapping[name]['mirrored']
        assert rec['group_id'] == rec['source_id'] == source
        assert seen_groups.setdefault(source,rec['split']) == rec['split']
        official = [k for k,v in original.items() if name in v]
        assert official == [rec['official_split']] == [item['official_split']]
        assert item['auxiliary_split'] == rec['split']
        with zipfile.ZipFile(archive) as z: text = z.read('texts/'+name+'.txt')
        assert hashlib.sha256(text).hexdigest() == item['caption_member_sha256']
        lines = [line.split('#') for line in text.decode('utf8').splitlines()]
        whole = [x[0] for x in lines if len(x)==4 and x[2:] == ['0.0','0.0']]
        assert rec['caption'] == whole[0] and len(whole) == item['whole_caption_count']
        assert len(lines)-len(whole) == item['other_caption_count']
        assert [x[2:] for x in lines] == item['raw_tags']
        path = HERE / rec['feature_path']
        assert path.name == name+'.npz' and sha(path) == rec['feature_sha256'] == item['feature_sha256']
        with np.load(path,allow_pickle=False) as d:
            x = np.asarray(d['features'],dtype=np.float32); t = d['time_grid'].copy()
        assert x.shape == (item['frames'],263) and len(t) == len(x)+1
        assert np.isfinite(x).all() and np.isin(x[:,-4:],[0.,1.]).all()
        np.testing.assert_allclose(np.diff(t),.05,rtol=0,atol=1e-14)
        if rec['split'] == 'train':
            train_features.append(x.astype(np.float64))
            with torch.no_grad(): points = recover_from_ric(torch.from_numpy(x),22).numpy()
            train_positions.append(points.reshape(len(x),66).astype(np.float64))
    feature = np.concatenate(train_features); spatial = np.concatenate(train_positions)
    assert len(feature) == report['train_frames'] == 86
    expected_stats = {'feature_mean':feature.mean(axis=0), 'feature_std':np.maximum(feature.std(axis=0),.001),
                      'spatial_mean':spatial.mean(axis=0), 'spatial_std':np.maximum(spatial.std(axis=0),.001)}
    errors = {}
    with np.load(stats_file,allow_pickle=False) as data:
        for key, value in expected_stats.items():
            errors[key] = float(np.max(np.abs(data[key]-value)))
            np.testing.assert_allclose(data[key],value,rtol=1e-10,atol=1e-12)
    result = {'report_sha256':sha(HERE/'report.json'),'manifest_sha256':sha(manifest_file),
              'stats_sha256':sha(stats_file),'records_verified':len(records),
              'train_frames_recomputed':len(feature),'stats_max_errors':errors,
              'official_vs_auxiliary_difference':['013303: official train; auxiliary val'],
              'training_ready':False,
              'limitations':'Technical checks only; decoder shared, no independent FK/encoding, semantic alignment, source identity, license, physics or subject-level independence.'}
    with (HERE/'independent_audit.json').open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
