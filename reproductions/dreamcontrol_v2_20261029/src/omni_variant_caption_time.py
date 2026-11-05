"""Catalog candidate captions and temporal coverage for all Omni variants.

This computes observable source-motion differences after the 10-second crop.
It does not infer whether a sentence is true, whether a box is in contact, or
whether any candidate is eligible for training.
"""
import collections
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
PAIRS=ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl'
ANNOT=ROOT/'omomo_text_pairing_20261101/omomo_text_anno.zip'
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def temporal_summary(qpos, fps):
    q=np.asarray(qpos,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=43 or len(q)<2 or not np.isfinite(q).all() or not np.isfinite(fps) or fps!=30:
        raise ValueError('invalid source qpos or declared fps')
    n=min(len(q),300)
    # Last point on the nominal 20 Hz grid, with integer rational arithmetic.
    target_index=((n-1)*20)//30
    frame_position=target_index*30/20
    lower=int(frame_position);fraction=frame_position-lower
    if lower+int(fraction>0)>=n:raise ValueError('target outside source crop')
    centers=q[:,40:43]
    at_end=centers[lower] if fraction==0 else centers[lower]*(1-fraction)+centers[lower+1]*fraction
    later=centers[lower+1:] if fraction else centers[lower:]
    distances=np.linalg.norm(later-at_end,axis=1)
    # This is only a time-support comparison; it does not use the actual FIR.
    return {'source_frames':len(q),'crop_frames':n,'source_tail_frames_after_crop':len(q)-n,
            'nominal_feature_rows':target_index,'target_last_sample_s':target_index/20.,
            'target_last_source_frame_position':frame_position,'full_last_sample_s':(len(q)-1)/30.,
            'target_to_full_end_center_m':float(np.linalg.norm(centers[-1]-at_end)),
            'target_to_later_max_center_m':float(distances.max()) if len(distances) else 0.,
            'full_caption_valid_for_variant_or_window':False}


def main():
    started=time.perf_counter();archive_hash=sha(ARCHIVE);pairs_hash=sha(PAIRS);annotation_hash=sha(ANNOT)
    if archive_hash!='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3' or pairs_hash!='96f30c547ad4b7a6fe8a0d24f71ac9b6ae016ce3cb14eafa7a67dd40d50fb69f':
        raise ValueError('source hash changed')
    rows=[json.loads(line) for line in PAIRS.read_text().splitlines()]
    variants=[row for row in rows if row['variant']!='original']
    if len(variants)!=1514 or len({row['motion_member'] for row in variants})!=1514:
        raise ValueError('variant coverage mismatch')
    captions=collections.Counter();trimmed=0;diff=[]
    with zipfile.ZipFile(ARCHIVE) as z,zipfile.ZipFile(ANNOT) as a,(HERE/'records.jsonl').open('x') as output:
        for row in variants:
            raw=z.read(row['motion_member']);caption_bytes=a.read(row['annotation_member'])
            if hashlib.sha256(raw).hexdigest()!=row['motion_member_sha256'] or hashlib.sha256(caption_bytes).hexdigest()!=row['annotation_member_sha256']:
                raise ValueError('source member or annotation changed')
            caption=json.loads(caption_bytes)[row['nominal_action_key']]
            if caption!=row['caption']:raise ValueError('caption mismatch')
            with np.load(io.BytesIO(raw),allow_pickle=False) as data:
                info=temporal_summary(data['qpos'],float(data['fps']))
            record={'motion_member':row['motion_member'],'motion_member_sha256':row['motion_member_sha256'],
                    'nominal_action_key':row['nominal_action_key'],'source_group':row['provisional_group'],
                    'annotation_member':row['annotation_member'],'annotation_sha256':row['annotation_member_sha256'],
                    'caption':caption,'caption_status':'name_matched_only_unreviewed','timeline':info,'training_ready':False}
            output.write(json.dumps(record,allow_nan=False)+'\n')
            captions[caption]+=1;trimmed+=int(info['source_tail_frames_after_crop']>0);diff.append(info['target_to_full_end_center_m'])
    if sha(ARCHIVE)!=archive_hash or sha(PAIRS)!=pairs_hash or sha(ANNOT)!=annotation_hash:
        raise RuntimeError('source changed')
    result={'archive_sha256':archive_hash,'pair_manifest_sha256':pairs_hash,'annotation_archive_sha256':annotation_hash,
            'records_sha256':sha(HERE/'records.jsonl'),'script_sha256':sha(Path(__file__)),
            'variants':len(variants),'cropped_before_source_end':trimmed,'caption_distribution':dict(captions),
            'target_to_full_end_center_m_quantiles':np.quantile(diff,[0,.5,.9,1]).tolist(),
            'elapsed_seconds':time.perf_counter()-started,'training_ready':False,
            'limitations':['Annotation has no per-frame timestamps and only a name-based relation to each variant.',
                           'Endpoint uses linear interpolation of raw object center, not the actual FIR.',
                           'Object movement does not prove any caption action or contact and is not quality acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
