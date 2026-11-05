"""Index legacy and independently reviewed sign-corrected technical previews.

This is not an accepted training manifest. No input file is changed or copied.
"""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data_loaders.humanml.scripts.motion_process import recover_from_ric

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'omomo_processed_full_preview_20261101';GROUP=ROOT/'omomo_group_split_20261101'
SOURCE=ROOT/'omni_variants_gpu_20261101';HERE=Path(__file__).resolve().parent
REPAIRS=(('omni_variants_signfix_snapshot_20261101','omni_variants_signfix_audit_20261101'),
         ('omni_variants_remaining_signfix_20261101','omni_variants_remaining_signfix_audit_20261101'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    base_report=json.loads((BASE/'report.json').read_text());base_audit=json.loads((BASE/'audit.json').read_text())
    if (sha(BASE/'manifest.json')!=base_report['manifest_sha256'] or base_audit['manifest_sha256']!=base_report['manifest_sha256'] or
        base_audit['preview_report_sha256']!=sha(BASE/'report.json')):raise ValueError('base preview mismatch')
    records=json.loads((BASE/'manifest.json').read_text())['records']
    for r in records:r['encoding_variant']='legacy'
    ids={r['sample_id'] for r in records};members={r['source_member'] for r in records}
    if len(ids)!=len(records) or len(members)!=len(records):raise ValueError('duplicate base rows')
    group_report=json.loads((GROUP/'report.json').read_text())
    if sha(GROUP/'candidates.jsonl')!=group_report['candidate_manifest_sha256']:raise ValueError('group input mismatch')
    groups={r['motion_member']:r for r in map(json.loads,(GROUP/'candidates.jsonl').read_text().splitlines())}
    full_report=json.loads((SOURCE/'report.json').read_text());full_audit=json.loads((SOURCE/'audit_v2.json').read_text())
    if sha(SOURCE/'records.jsonl')!=full_report['records_sha256'] or sha(SOURCE/'report.json')!=full_audit['report_sha256']:
        raise ValueError('full source mismatch')
    source={r['sample_id']:r for r in map(json.loads,(SOURCE/'records.jsonl').read_text().splitlines())}
    pins={}
    for repair_name,audit_name in REPAIRS:
        rp,ap=ROOT/repair_name,ROOT/audit_name
        rr=json.loads((rp/'report.json').read_text());aa=json.loads((ap/'report.json').read_text())
        if (sha(rp/'records.jsonl')!=rr['rows_sha256'] or aa['repair_report_sha256']!=sha(rp/'report.json') or
            sha(ap/'records.jsonl')!=aa['rows_sha256'] or aa['failed'] or aa['checked']!=rr['technical_previews']):
            raise ValueError('repair evidence mismatch')
        checked={x['sample_id'] for x in map(json.loads,(ap/'records.jsonl').read_text().splitlines()) if x['status']=='checked'}
        rows=[x for x in map(json.loads,(rp/'records.jsonl').read_text().splitlines()) if x['status']=='technical_preview']
        if checked!={x['sample_id'] for x in rows} or len(rows)!=rr['technical_previews']:raise ValueError('repair coverage mismatch')
        pins[repair_name]={'repair_report_sha256':sha(rp/'report.json'),'audit_report_sha256':sha(ap/'report.json'),'audit_folder':audit_name}
        for item in rows:
            r=source[item['sample_id']];g=groups[r['source_member']];feature=rp/item['feature'];state=SOURCE/r['state20']
            if (r['sample_id'] not in full_audit['failed'] or r['sample_id'] in ids or r['source_member'] in members or item['parent_state_sha256']!=r['state20_sha256'] or
                item['parent_feature_sha256']!=r['feature_sha256'] or sha(state)!=r['state20_sha256'] or sha(feature)!=item['feature_sha256'] or
                g['motion_member_sha256']!=r['source_sha256'] or g['annotation_sha256']!=r['annotation_sha256'] or g['caption']!=r['caption']):
                raise ValueError('repair source, feature or duplicate mismatch')
            ids.add(r['sample_id']);members.add(r['source_member'])
            records.append({'sample_id':r['sample_id'],'source_member':r['source_member'],'source_sha256':r['source_sha256'],
                'nominal_subject':g['nominal_subject'],'nominal_action_key':g['nominal_action_key'],
                'auxiliary_split':g['auxiliary_split'],'caption':r['caption'],'caption_status':'unreviewed_filename_match',
                'feature_path':str(feature.relative_to(ROOT)),'feature_sha256':item['feature_sha256'],
                'state_path':str(state.relative_to(ROOT)),'state_sha256':r['state20_sha256'],
                'feature_frames':r['feature_frames'],'processing_evidence_folder':repair_name,
                'parent_feature_sha256':r['feature_sha256'],'encoding_variant':'sign_continuous_v1','training_ready':False})
    counts=collections.Counter();frames=collections.Counter();fsum=np.zeros(263);fsq=np.zeros(263);psum=np.zeros(66);psq=np.zeros(66);n=0
    for r in records:
        feature=(ROOT/r['feature_path']).resolve(strict=True);state=(ROOT/r['state_path']).resolve(strict=True)
        if not feature.is_relative_to(ROOT.resolve()) or not state.is_relative_to(ROOT.resolve()) or sha(feature)!=r['feature_sha256'] or sha(state)!=r['state_sha256']:
            raise ValueError('preview file hash mismatch')
        with np.load(feature,allow_pickle=False) as f:x=np.asarray(f['features'],dtype=np.float32)
        if x.shape!=(r['feature_frames'],263) or not np.isfinite(x).all():raise ValueError('invalid feature')
        split=r['auxiliary_split'];counts[split]+=1;frames[split]+=len(x)
        if split=='train':
            a=x.astype(np.float64)
            with torch.no_grad(): p=recover_from_ric(torch.from_numpy(x),22).numpy().reshape(-1,66).astype(np.float64)
            if not np.isfinite(p).all():raise ValueError('invalid decoded points')
            fsum+=a.sum(0);fsq+=(a*a).sum(0);psum+=p.sum(0);psq+=(p*p).sum(0);n+=len(x)
    if set(counts)!={'train','val','test'} or n==0:raise ValueError('missing split')
    floor=.001;fm=fsum/n;pm=psum/n
    with (HERE/'stats.npz').open('xb') as f:np.savez_compressed(f,feature_mean=fm,feature_std=np.maximum(np.sqrt(np.maximum(fsq/n-fm*fm,0)),floor),spatial_mean=pm,spatial_std=np.maximum(np.sqrt(np.maximum(psq/n-pm*pm,0)),floor))
    with (HERE/'manifest.json').open('x') as f:json.dump({'records':records,'training_ready':False},f,indent=2,allow_nan=False);f.write('\n')
    result={'base_manifest_sha256':base_report['manifest_sha256'],'base_audit_sha256':sha(BASE/'audit.json'),
        'group_manifest_sha256':group_report['candidate_manifest_sha256'],'repair_pins':pins,
        'manifest_sha256':sha(HERE/'manifest.json'),'stats_sha256':sha(HERE/'stats.npz'),'script_sha256':sha(__file__),
        'counts_by_split':dict(counts),'feature_frames_by_split':dict(frames),'encoding_counts':dict(collections.Counter(r['encoding_variant'] for r in records)),
        'min_std':floor,'training_ready':False,
        'limitations':['Sign-continuous encoding is an explicit project approximation, not author v2.',
            'FK/encoder and spatial decoder are shared with previous checks.',
            'Caption, source rights, true time, physical quality and training acceptance remain unverified.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
