"""Create a non-training six-candidate technical data/normalization preview.

The BML captions are unvalidated full-clip annotations of shorter excerpts.
They are included solely to test loader shape and traceability, and every
record explicitly says caption_accepted=False. No training is authorized.
"""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from audit_robot_features import audit
from training_stats import training_stats

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    viewpath=ROOT/'source_grouping_v3_20261031/six_source_preview.json'
    checkpath=ROOT/'source_grouping_v3_20261031/independent_audit.json'
    view=json.loads(viewpath.read_text()); check=json.loads(checkpath.read_text())
    if check['status']!='passed' or sha(viewpath)!=check['preview_sha256']:
        raise ValueError('grouped candidate audit mismatch')
    olddir=ROOT/'ssm_four_preview_20261031'
    old=json.loads((olddir/'experimental_manifest_not_for_training.json').read_text())
    oldmap={r['sample_id']:r for r in old['records']}
    bml=ROOT/'bml_joint_smoothness_features_20261031'
    records=[]; arrays=[]; detail=[]
    for item in view['records']:
        sample=item['sample_id']
        if item['kind']=='existing_ssm_preview':
            rec=oldmap[sample];src=olddir/rec['feature_path']; expected=rec['feature_sha256']
            caption=rec['caption'];caption_source=rec['caption_source']
            annotation={'caption':caption,'source':caption_source,'accepted':False,
                        'reason':'full-clip semantic correspondence not reviewed'}
        elif item['kind']=='bml_pending_robot_output':
            label=sample[4:]; src=bml/label/'candidate.npz'
            report=json.loads((bml/label/'report.json').read_text())
            checked=json.loads((bml/label/'audit_report.json').read_text())
            expected=report['feature_sha256']
            if checked['feature_sha256']!=expected or checked['candidate_report_sha256']!=sha(bml/label/'report.json'):
                raise ValueError('BML feature audit mismatch')
            selected=item['candidate_clip_annotations'][0]
            entry=selected['annotations'][0];caption=entry['caption']
            caption_source='HumanML texts.zip: texts/'+selected['clip_id']+'.txt'
            annotation={'caption':caption,'source':caption_source,'accepted':False,
                        'raw_tags':[entry['start_raw'],entry['end_raw']],
                        'excerpt_first_last_seconds_relative_to_clip':selected['excerpt_first_last_seconds_relative_to_clip'],
                        'source_member_sha256':selected['caption_member_sha256'],
                        'reason':'whole-clip annotation; excerpt semantics unverified'}
        else:raise ValueError('unknown candidate')
        if sha(src)!=expected or not caption:raise ValueError('input changed or caption missing')
        dst=HERE/(sample+'.npz')
        with src.open('rb') as source, dst.open('xb') as target:shutil.copyfileobj(source,target)
        if sha(dst)!=expected:raise ValueError('copy mismatch')
        with np.load(dst,allow_pickle=False) as data:
            features=data['features'].copy()
        records.append({'sample_id':sample,'source_id':item['source_id'],
                        'group_id':item['provenance_keys']['source_directory'][0],
                        'provenance_keys':item['provenance_keys'],'split':item['split'],
                        'caption':caption,'caption_source':caption_source,'caption_accepted':False,
                        'feature_path':dst.name,'feature_sha256':expected,'fps':20,'training_ready':False})
        arrays.append((item['split'],features));detail.append({'sample_id':sample,'annotation':annotation,'frames':len(features)})
    stats=training_stats(arrays,min_std=.001)
    joints=[]
    for split,x in arrays:
        if split=='train':
            with torch.no_grad():j=recover_from_ric(torch.from_numpy(x),22).numpy()
            joints.append(j.reshape(len(j),66).astype(np.float64))
    joints=np.concatenate(joints)
    values={'feature_mean':stats['mean'],'feature_std':stats['std'],
            'spatial_mean':joints.mean(0),'spatial_std':np.maximum(joints.std(0),.001)}
    with (HERE/'experimental_stats_not_for_training.npz').open('xb') as f:np.savez_compressed(f,**values)
    manifest={'records':records,'training_ready':False,'purpose':'unaccepted technical preview',
              'limitations':['captions unverified, source identity and physical quality not accepted',
                             'auxiliary grouping is not official benchmark split',
                             'two train motions from one source directory']}
    with (HERE/'experimental_manifest_not_for_training.json').open('x',encoding='utf8') as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    result=audit(HERE,records,values,min_std=.001)
    if result['training_ready'] is not False or result['explicit_provenance_groups_disjoint'] is not True:
        raise ValueError('unexpected audit result')
    out={'view_sha256':sha(viewpath),'view_audit_sha256':sha(checkpath),'details':detail,
         'manifest_sha256':sha(HERE/'experimental_manifest_not_for_training.json'),
         'stats_sha256':sha(HERE/'experimental_stats_not_for_training.npz'),
         'train_frames':stats['frame_count'],'clamped_feature_dimensions':len(stats['clamped_channels']),
         'feature_audit':result,'script_sha256':sha(Path(__file__)),'training_ready':False,
         'limitations':manifest['limitations']}
    with (HERE/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'train_frames':stats['frame_count'],'counts':result['counts'],'training_ready':False}),flush=True)


if __name__=='__main__':main()
