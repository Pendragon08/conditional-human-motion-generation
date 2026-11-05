"""Independent bookkeeping/statistics audit of an unaccepted preview."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent;BASE=ROOT/'mixed_manual_caption_preview_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    build=json.loads((HERE/'build_report.json').read_text())
    records=json.loads((HERE/'manifest.json').read_text())['records']
    states=json.loads((HERE/'state_manifest.json').read_text())['records']
    ann=json.loads((HERE/'annotation.json').read_text());pre=json.loads((HERE/'preflight.json').read_text())
    assert sha(BASE/'manifest.json')==build['base_manifest_sha256']
    assert sha(HERE/'manifest.json')==build['manifest_sha256'] and sha(HERE/'state_manifest.json')==build['state_manifest_sha256']
    assert sha(HERE/'stats.npz')==build['stats_sha256']
    assert records[:5]==json.loads((BASE/'manifest.json').read_text())['records']
    new=records[-1];assert new['sample_id']=='007410' and new['caption_origin']=='manual_visual'
    assert new['caption']==ann['caption'] and new['caption_review_sha256']==sha(HERE/'annotation.json')
    assert all(sha(HERE/name)==h for name,h in ann['review_images_sha256'].items())
    by_state={s['sample_id']:s for s in states};features=[];spatial=[];groups={}
    for row in records:
        state=by_state[row['sample_id']];p=HERE/row['feature_path'];s=HERE/state['state_path']
        assert sha(p)==row['feature_sha256'] and sha(s)==state['state_sha256']
        assert groups.setdefault(row['group_id'],row['split'])==row['split']
        with np.load(p,allow_pickle=False) as f:x=f['features'].copy()
        if row['split']=='train':
            features.append(x.astype(np.float64))
            with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(float))
    x=np.concatenate(features);p=np.concatenate(spatial)
    with np.load(HERE/'stats.npz',allow_pickle=False) as s:
        errors=[float(np.max(abs(s['feature_mean']-x.mean(0)))),float(np.max(abs(s['feature_std']-np.maximum(x.std(0),.001)))),
                float(np.max(abs(s['spatial_mean']-p.mean(0)))),float(np.max(abs(s['spatial_std']-np.maximum(p.std(0),.001))))]
    assert len(x)==236==build['train_feature_frames'] and max(errors)<1e-6
    with np.load(HERE/'007410.npz',allow_pickle=False) as f:
        times=f['sample_times'];raw=f['raw_source_frames']
    rate=59.9999885559082
    assert np.allclose(np.array(ann['sample_source_frame_positions'])/rate,times,atol=1e-8,rtol=0)
    assert ann['source_frame_bounds']==[int(raw.min()),int(raw.max())] and len(times)==78
    assert pre['feature_audit']['counts']=={'train':4,'val':1,'test':1}
    assert pre['state_audit']['state_technical_checks_passed'] is True
    assert len(pre['caption_archive_binding']['cases'])==4 and len(pre['manual_caption_binding']['cases'])==2
    assert pre['ssm_source_evidence_audit']['marker_evidence_cases']==3
    assert pre['feature_audit']['training_ready'] is False
    assert 'training blocked: source and applicable quality acceptance not yet implemented' in (HERE/'run_rejected.log').read_text()
    series=ROOT/'ssm_007410_punch_sequence_v2_20261101/series.npz'
    with np.load(series,allow_pickle=False) as f:
        values={name:{'peak_toe_axis_m':float(f[name+'_right_wrist_initial_toe_axis_m'].max()),
            'start_relative_shoulder_m':float(f[name+'_right_wrist_height_relative_shoulder_m'][0]),
            'peak_relative_shoulder_m':float(f[name+'_right_wrist_height_relative_shoulder_m'].max()),
            'end_relative_shoulder_m':float(f[name+'_right_wrist_height_relative_shoulder_m'][-1])}
            for name in ('human','robot')}
    out={'manifest_sha256':sha(HERE/'manifest.json'),'annotation_sha256':sha(HERE/'annotation.json'),
         'stats_sha256':sha(HERE/'stats.npz'),'preflight_sha256':sha(HERE/'preflight.json'),
         'script_sha256':sha(Path(__file__)),'train_feature_frames':len(x),'stats_max_abs_differences':errors,
         'split_counts':pre['feature_audit']['counts'],'manual_caption_count':2,
         'visual_support_previously_computed_series':values,'training_ready':False,
         'limitations':['Same FK/decoder and existing visual evidence; no independent semantic reviewer.',
             'Toe axis assumes forward; no real capture clock, physical or license acceptance.',
             'Tiny unaccepted auxiliary preview, not an adequate training/evaluation set.']}
    with (HERE/'audit.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
