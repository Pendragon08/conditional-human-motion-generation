"""Recompute whole-trajectory object-center statistics with scalar loops."""
import hashlib,io,json,math,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent;ZIP=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
PAIRS=ROOT/'omomo_text_pairing_20261101/candidate_pairs.jsonl'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    report=json.loads((HERE/'report.json').read_text());rows=[json.loads(x) for x in (HERE/'frames_summary.jsonl').read_text().splitlines()]
    candidates=[json.loads(x) for x in PAIRS.read_text().splitlines()]
    candidates={x['nominal_action_key']:x for x in candidates if x['variant']=='original'}
    assert sha(ZIP)==report['archive_sha256'] and sha(PAIRS)==report['pairs_sha256']
    assert sha(HERE/'frames_summary.jsonl')==report['rows_sha256'] and len(rows)==len(candidates)==312
    seen=set();truncations=Counter();low_drop=Counter();values=defaultdict(list);quat_bad=0
    with zipfile.ZipFile(ZIP) as z:
        for row in rows:
            sid=row['sample_id'];assert sid not in seen and sid in candidates;seen.add(sid);p=candidates[sid]
            assert row['caption']==p['caption'] and row['motion_member']==p['motion_member']
            raw=z.read(row['motion_member']);assert hashlib.sha256(raw).hexdigest()==row['motion_member_sha256']==p['motion_member_sha256']
            with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                q=f['qpos'];fps=float(f['fps'])
            heights=[float(q[i,42]) for i in range(len(q))]
            peak=max(range(len(heights)),key=heights.__getitem__)
            h=[math.hypot(float(q[i,40]-q[0,40]),float(q[i,41]-q[0,41])) for i in range(len(q))]
            err=max(abs(math.sqrt(sum(float(q[i,j])**2 for j in range(36,40)))-1.) for i in range(len(q)))
            checks={'object_z_start_m':heights[0],'object_z_peak_m':heights[peak],'object_z_end_m':heights[-1],
                    'rise_from_start_m':heights[peak]-heights[0],'fall_to_end_m':heights[peak]-heights[-1],
                    'max_center_horizontal_displacement_m':max(h),'end_center_horizontal_displacement_m':h[-1],
                    'object_quaternion_norm_max_abs_error':err,'nominal_last_sample_s':(len(q)-1)/fps,
                    'peak_nominal_seconds':peak/fps}
            assert all(abs(row[k]-v)<1e-10 for k,v in checks.items()) and row['peak_frame']==peak and row['frames']==len(q)
            assert row['semantic_reviewed'] is False and row['training_ready'] is False
            # The current experimental encoder caps input at 200 samples, so its
            # last available point at 20 Hz is 9.95 s, not 10.00 s.
            if checks['nominal_last_sample_s']>9.95+1e-9:truncations[row['caption']]+=1
            if checks['rise_from_start_m']>1e-9 and checks['fall_to_end_m']<.5*checks['rise_from_start_m']:
                low_drop[row['caption']]+=1
            if err>1e-4:quat_bad+=1
            values[row['caption']].append(row)
    assert seen==set(candidates) and quat_bad==report['counts'].get('bad_object_quaternion_norm',0)
    assert sum(r['nominal_last_sample_s']>10 for r in rows)==report['counts']['exceeds_ten_nominal_seconds']
    result={'source_archive_sha256':sha(ZIP),'pairs_sha256':sha(PAIRS),'rows_sha256':sha(HERE/'frames_summary.jsonl'),
            'originals_audited':312,'source_last_sample_beyond_9_95s':sum(truncations.values()),
            'beyond_9_95s_by_caption':dict(truncations),'fall_less_than_half_of_center_rise_by_caption':dict(low_drop),
            'object_quaternion_norm_violations':quat_bad,'training_ready':False,
            'limitations':['Numerical independent loop on same released robot/object data; no human or frame-time ground truth.',
                '9.95 s is current implementation endpoint, not an author-specific segmentation rule.',
                'Half-rise is an exploratory display criterion, not a semantic or quality threshold.',
                'Center height is not surface support; put-down may be on elevated surfaces.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
