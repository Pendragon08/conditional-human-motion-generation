"""Read whole original robot-object tracks and describe object-center motion.

This ranks future caption review candidates, not semantic or physical truth.
The OmniRetarget Z-up/world-frame and 30 Hz conventions remain declarations.
No training samples, segmentation, threshold filter or robot fit is produced.
"""
import hashlib,io,json
from collections import Counter,defaultdict
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
PAIRS=ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl'
SOURCE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
PAIRS_SHA='96f30c547ad4b7a6fe8a0d24f71ac9b6ae016ce3cb14eafa7a67dd40d50fb69f'
ARCHIVE_SHA='93001f298aec587cd2b0a79aa5553d264cf2148e5397d4aa4bb767a9d58041b3'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    if sha(PAIRS)!=PAIRS_SHA or sha(SOURCE)!=ARCHIVE_SHA:raise ValueError('source hash mismatch')
    pairs=[json.loads(s) for s in PAIRS.read_text().splitlines()]
    originals=[p for p in pairs if p['variant']=='original']
    if len(originals)!=312 or len({p['nominal_action_key'] for p in originals})!=312:
        raise ValueError('original key coverage mismatch')
    totals=Counter();groups=defaultdict(list)
    with zipfile.ZipFile(SOURCE) as archive,(HERE/'frames_summary.jsonl').open('x') as sink:
        for p in originals:
            raw=archive.read(p['motion_member'])
            if hashlib.sha256(raw).hexdigest()!=p['motion_member_sha256']:raise ValueError('member hash mismatch')
            with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                x=np.asarray(f['qpos'],dtype=np.float64);fps=float(f['fps'])
            if (x.ndim!=2 or x.shape[1]!=43 or len(x)<2 or not np.isfinite(x).all() or
                not np.isfinite(fps) or fps<=0):raise ValueError('invalid qpos or frame rate')
            norm=np.linalg.norm(x[:,36:40],axis=1)
            norm_error=float(np.max(np.abs(norm-1)))
            obj=x[:,40:43];v=obj[:,2];xy=obj[:,:2]-obj[0,:2]
            peak=int(np.argmax(v));displacement=np.linalg.norm(xy,axis=1)
            row={'sample_id':p['nominal_action_key'],'motion_member':p['motion_member'],
                 'motion_member_sha256':p['motion_member_sha256'],'annotation_member':p['annotation_member'],
                 'annotation_member_sha256':p['annotation_member_sha256'],'caption':p['caption'],
                 'subject_group':p['provisional_group'],'frames':len(x),'declared_fps':fps,
                 'nominal_last_sample_s':float((len(x)-1)/fps),'object_quaternion_norm_max_abs_error':norm_error,
                 'object_z_start_m':float(v[0]),'object_z_peak_m':float(v[peak]),'object_z_end_m':float(v[-1]),
                 'peak_frame':peak,'peak_nominal_seconds':float(peak/fps),
                 'rise_from_start_m':float(v[peak]-v[0]),'fall_to_end_m':float(v[peak]-v[-1]),
                 'max_center_horizontal_displacement_m':float(np.max(displacement)),
                 'end_center_horizontal_displacement_m':float(displacement[-1]),
                 'candidate_pair_only':True,'semantic_reviewed':False,'training_ready':False}
            sink.write(json.dumps(row,allow_nan=False)+'\n');totals['originals']+=1
            if norm_error>1e-4:totals['bad_object_quaternion_norm']+=1
            if row['nominal_last_sample_s']>10:totals['exceeds_ten_nominal_seconds']+=1
            groups[p['caption']].append(row)
    if sha(PAIRS)!=PAIRS_SHA or sha(SOURCE)!=ARCHIVE_SHA:raise RuntimeError('source changed')
    summary=[]
    for caption,items in sorted(groups.items()):
        summary.append({'caption':caption,'count':len(items),
            'max_rise_median_m':float(np.median([x['rise_from_start_m'] for x in items])),
            'fall_to_end_median_m':float(np.median([x['fall_to_end_m'] for x in items])),
            'horizontal_max_median_m':float(np.median([x['max_center_horizontal_displacement_m'] for x in items]))})
    report={'pairs_sha256':PAIRS_SHA,'archive_sha256':ARCHIVE_SHA,'rows_sha256':sha(HERE/'frames_summary.jsonl'),
            'script_sha256':sha(Path(__file__)),'counts':dict(totals),'caption_summaries':summary,
            'training_ready':False,'limitations':['Object center motion is not object-surface contact or semantic truth.',
             'Z-up and source FPS are declared conventions, not measured clock or independently calibrated ground.',
             'Entire source caption may not apply to a 10-second subwindow or an augmented version.',
             'Identity, caption semantics, license, 29-to-27 conversion and physical feasibility remain unaccepted.']}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
