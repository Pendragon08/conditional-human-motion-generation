"""Triage object-motion evidence for candidate text without accepting captions.

All frames in each available processed state are inspected.  These engineering
thresholds are review aids, not author v2 standards or physical acceptance.
Object center and orientation alone cannot establish contact, force, kick,
push, pull, release, or final support surface.
"""
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'omomo_processed_snapshot_preview_20261101';HERE=Path(__file__).resolve().parent
POSITION_EPS_M=.02
ANGLE_EPS_RAD=float(np.deg2rad(5.))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def observed_motion(obj,times):
    obj=np.asarray(obj,dtype=np.float64);times=np.asarray(times,dtype=np.float64)
    if obj.ndim!=2 or obj.shape[1]!=7 or len(obj)<2 or times.shape!=(len(obj),) or not np.isfinite(obj).all() or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('invalid object pose or timeline')
    q=obj[:,:4];norm=np.linalg.norm(q,axis=1)
    if np.any(norm<1e-8) or not np.allclose(norm,1.,atol=1e-5,rtol=0):raise ValueError('invalid object quaternion')
    q=q/norm[:,None];p=obj[:,4:];z=p[:,2]
    rise=z-np.minimum.accumulate(z);fall=np.maximum.accumulate(z)-z
    travel=np.linalg.norm(p-p[0],axis=1)
    angles=2*np.arccos(np.clip(np.abs(q@q[0]),0.,1.))
    return {'frames':len(p),'nominal_start_s':float(times[0]),'nominal_end_s':float(times[-1]),
            'ordered_max_rise_m':float(rise.max()),'ordered_max_fall_m':float(fall.max()),
            'max_translation_from_start_m':float(travel.max()),'max_rotation_from_start_rad':float(angles.max()),
            'net_vertical_m':float(z[-1]-z[0]),'end_from_max_height_m':float(z.max()-z[-1])}


def review_flags(caption,metrics):
    text=caption.lower();flags=[]
    if 'lift' in text and metrics['ordered_max_rise_m']<POSITION_EPS_M:flags.append('small_observed_center_rise')
    if ('put down' in text or 'set it back down' in text or 'set the largebox back down' in text) and metrics['ordered_max_fall_m']<POSITION_EPS_M:
        flags.append('small_observed_center_fall')
    if 'move' in text and metrics['max_translation_from_start_m']<POSITION_EPS_M:flags.append('small_observed_center_translation')
    if 'rotate' in text and metrics['max_rotation_from_start_rad']<ANGLE_EPS_RAD:flags.append('small_observed_object_rotation')
    if any(term in text for term in ('push','pull','kick','release','lift','put down','set it back down')):
        flags.append('contact_force_and_support_not_observed')
    return flags


def main():
    report=json.loads((PREVIEW/'report.json').read_text());audit=json.loads((PREVIEW/'audit.json').read_text())
    manifest=PREVIEW/'manifest.json';mh=sha(manifest)
    if mh!=report['manifest_sha256'] or audit['manifest_sha256']!=mh or audit['preview_report_sha256']!=sha(PREVIEW/'report.json'):
        raise ValueError('preview evidence changed')
    records=json.loads(manifest.read_text())['records'];results=[];flag_counts=collections.Counter();caption_counts=collections.Counter()
    with (HERE/'records.jsonl').open('x',buffering=1) as out:
        for row in records:
            path=(ROOT/row['state_path']).resolve(strict=True)
            if not path.is_relative_to(ROOT.resolve()) or sha(path)!=row['state_sha256']:raise ValueError('state pin mismatch')
            with np.load(path,allow_pickle=False) as f:metrics=observed_motion(f['object_wxyz_xyz'],f['sample_times'])
            if metrics['frames']!=row['feature_frames']+1:raise ValueError('state feature length mismatch')
            flags=review_flags(row['caption'],metrics)
            if sha(path)!=row['state_sha256']:raise RuntimeError('state changed during triage')
            result={'sample_id':row['sample_id'],'state_path':row['state_path'],'state_sha256':row['state_sha256'],
                    'caption':row['caption'],'caption_status':'unreviewed_filename_match',
                    'auxiliary_split':row['auxiliary_split'],'metrics':metrics,'review_flags':flags,'training_ready':False}
            results.append(result);out.write(json.dumps(result,allow_nan=False)+'\n')
            flag_counts.update(flags);caption_counts[row['caption']]+=1
    if sha(manifest)!=mh:raise RuntimeError('preview manifest changed')
    result={'preview_manifest_sha256':mh,'preview_audit_sha256':sha(PREVIEW/'audit.json'),
            'script_sha256':sha(Path(__file__)),'records_sha256':sha(HERE/'records.jsonl'),
            'samples':len(results),'position_review_threshold_m':POSITION_EPS_M,'angle_review_threshold_rad':ANGLE_EPS_RAD,
            'flag_counts':dict(flag_counts),'caption_counts':dict(caption_counts),'training_ready':False,
            'limitations':['Thresholds are chosen engineering review aids, not author or physical criteria.',
                           'Center/rotation proxies neither prove nor disprove a textual action.',
                           'Actual contact, hand release, push/pull/kick, surface and source identity are not observed.',
                           'FIR pose and nominal timestamps are approximate, not measured clock or physical acceptance.']}
    with (HERE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
