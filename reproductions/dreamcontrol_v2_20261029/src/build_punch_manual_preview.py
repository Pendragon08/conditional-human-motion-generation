"""Create an isolated, unaccepted six-sample technical preview.

007410's text is a new subjective visual annotation, not HumanML text. This
script consumes previously generated data; it performs no retargeting or training.
"""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from training_stats import training_stats
from provenance_groups import audit_provenance

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'mixed_manual_caption_preview_20261101'
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'ssm_anchored_20hz_candidates_20261031/007410.npz'
SOURCE_HASH='d39dd557900293aec37b33b9c86561c2339d62fca7ba37dd5ff9156c079196fa'
RATE=59.9999885559082
CAPTION='a person extends their right arm forward and then lowers it.'
IMAGES={
 '007410_02_000.png':'d778d333f46ed9a590e26f28aacbce22584ad5b831f59023d0d8654f4ec477b8',
 '007410_02_040.png':'30cfb34e6a59a6fb1380a2c9fb2175a924181b037110a808d51badb3f3f112c6',
 '007410_12_000.png':'4fa8013c76be802718772af0d79358aab873b1a3a8ef4fdcaf4f4235cd1a0efb',
 '007410_12_040.png':'e001beb06f6d2bc0d905ca75bace2f7f36abe4e0dedd592e839dd0432656a199'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def copy(src,dst,expected):
    if sha(src)!=expected:raise ValueError('source artifact hash mismatch')
    if dst.exists():
        if sha(dst)!=expected:raise ValueError('conflicting duplicate artifact')
        return
    with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
    if sha(dst)!=expected:raise ValueError('copy changed')

def main():
    torch.set_num_threads(1)
    records=json.loads((BASE/'manifest.json').read_text())['records']
    states=json.loads((BASE/'state_manifest.json').read_text())['records']
    cfg=json.loads((BASE/'config.json').read_text())
    alignment=[json.loads(x) for x in Path(cfg['ssm_alignment_path']).read_text().splitlines()]
    match=[a for a in alignment if a.get('motion_id')=='007410' and a.get('mirrored') is False]
    if len(match)!=1 or match[0]['reference_fps']!=RATE:raise ValueError('source mapping mismatch')
    if sha(SOURCE)!=SOURCE_HASH:raise ValueError('candidate changed')
    for image,expected in IMAGES.items():
        if sha(HERE/image)!=expected:raise ValueError('review image hash mismatch')
    with np.load(SOURCE,allow_pickle=False) as f:
        times=np.asarray(f['sample_times'],dtype=np.float64)
        features=np.asarray(f['features'])
        raw=np.asarray(f['raw_source_frames'])
    if features.shape!=(77,263) or times.shape!=(78,) or not np.allclose(np.diff(times),.05,rtol=0,atol=1e-8):
        raise ValueError('unexpected candidate shape or timeline')
    if int(raw.min())!=1 or int(raw.max())!=232:raise ValueError('unexpected source frame bounds')
    positions=(times*RATE).tolist()
    annotation={'sample_id':'007410','caption':CAPTION,'feature_sha256':SOURCE_HASH,
        'caption_origin':'manual visual annotation; not provided by HumanML or original dataset',
        'reviewer':'Codex','semantic_supported_for_exported_window':True,'training_ready':False,
        'timeline_mapping':'linear_source_frame_positions','sample_source_frame_positions':positions,
        'source_frame_bounds':[1,232],'feature_rows':77,'nominal_time_s':[float(times[0]),float(times[-1])],
        'review_images_sha256':IMAGES,'reason':'Viewed both 78-frame projections of fitted human and approximate G1; right arm extends then lowers.',
        'limitations':['Forward is interpreted from the original toe-axis proxy, not independently calibrated.',
            'Subjective visual annotation, not a source caption or independent second review.',
            'FIR and interpolation context, true clock, identity, contact and physics not accepted.']}
    save(HERE/'annotation.json',annotation)
    for row in records:copy(BASE/row['feature_path'],HERE/row['feature_path'],row['feature_sha256'])
    for row in states:copy(BASE/row['state_path'],HERE/row['state_path'],row['state_sha256'])
    copy(SOURCE,HERE/'007410.npz',SOURCE_HASH)
    save(HERE/'annotation_audit.json',{'annotation_sha256':sha(HERE/'annotation.json'),
        'candidate_sha256':sha(SOURCE),'candidate_copy_sha256':sha(HERE/'007410.npz'),
        'source_alignment_sha256':sha(Path(cfg['ssm_alignment_path'])),
        'source_receipt_sha256':sha(Path(match[0]['source_receipt'])),
        'review_images_sha256':{name:sha(HERE/name) for name in IMAGES},
        'training_ready':False,'limitations':['Hashes and time mapping checked; no independent semantic reviewer.']})
    records.append({'sample_id':'007410','source_id':match[0]['source_path'],
        'group_id':'conservative-family:SSM_synced',
        'provenance_keys':{'source_path':[match[0]['source_path']],
            'source_directory':['pose_data/SSM_synced/20160330_03333'],'source_family':['SSM_synced']},
        'split':'train','caption':CAPTION,'caption_source':'Codex manual visual annotation; not source text',
        'caption_origin':'manual_visual','caption_accepted':True,'caption_review_sha256':sha(HERE/'annotation.json'),
        'feature_path':'007410.npz','feature_sha256':SOURCE_HASH,'fps':20,'training_ready':False,
        'auxiliary_split_note':'same source family and likely recording subject as 011120; not an independent identity'})
    states.append({'sample_id':'007410','feature_sha256':SOURCE_HASH,'state_path':'007410.npz',
                   'state_sha256':SOURCE_HASH,'foot_velocity_squared_threshold':.002})
    if not audit_provenance(records)['source_groups_disjoint']:raise ValueError('source group crosses split')
    train=[];spatial=[]
    for row in records:
        if row['split']!='train':continue
        with np.load(HERE/row['feature_path'],allow_pickle=False) as f:x=f['features'].copy()
        train.append(('train',x))
        with torch.no_grad():spatial.append(recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(float))
    stats=training_stats(train,min_std=.001);points=np.concatenate(spatial)
    with (HERE/'stats.npz').open('xb') as f:
        np.savez_compressed(f,feature_mean=stats['mean'],feature_std=stats['std'],
                            spatial_mean=points.mean(0),spatial_std=np.maximum(points.std(0),.001))
    save(HERE/'manifest.json',{'records':records,'training_ready':False})
    save(HERE/'state_manifest.json',{'records':states,'training_ready':False})
    manual=json.loads((BASE/'manual_caption_evidence.json').read_text())
    manual['receipts'].append({'sample_id':'007410','annotation_path':HERE.name+'/annotation.json',
        'annotation_sha256':sha(HERE/'annotation.json'),'audit_path':HERE.name+'/annotation_audit.json',
        'audit_sha256':sha(HERE/'annotation_audit.json'),'declared_source_fps':RATE})
    save(HERE/'manual_caption_evidence.json',manual)
    evidence=json.loads(Path(cfg['ssm_source_evidence_manifest']).read_text())
    evidence['receipts'].append({'sample_id':'007410','receipt_sha256':sha(Path(match[0]['source_receipt']))})
    save(HERE/'source_evidence_manifest.json',evidence)
    cfg.update({'dataset_root':str(HERE),'manifest':str(HERE/'manifest.json'),'stats':str(HERE/'stats.npz'),
        'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json'),'run_dir':str(HERE),
        'ssm_source_evidence_manifest':str(HERE/'source_evidence_manifest.json'),
        'purpose':'unaccepted six-candidate mixed-caption technical preview',
        'manual_caption_binding':{'evidence_root':str(ROOT),'evidence_manifest':str(HERE/'manual_caption_evidence.json'),
                                  'state_root':str(HERE),'state_manifest':str(HERE/'state_manifest.json')}})
    for key in ('caption_archive_binding','vertical_support_diagnostic'):
        cfg[key]['state_root']=str(HERE);cfg[key]['state_manifest']=str(HERE/'state_manifest.json')
    save(HERE/'config.json',cfg)
    save(HERE/'build_report.json',{'base_manifest_sha256':sha(BASE/'manifest.json'),
        'manifest_sha256':sha(HERE/'manifest.json'),'state_manifest_sha256':sha(HERE/'state_manifest.json'),
        'stats_sha256':sha(HERE/'stats.npz'),'train_feature_frames':int(stats['frame_count']),
        'script_sha256':sha(Path(__file__)),'training_ready':False})
if __name__=='__main__':main()
