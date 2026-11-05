"""Independently recompute all 718 object-motion triage records."""
import collections
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'omomo_processed_snapshot_preview_20261101';HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    report=json.loads((HERE/'report.json').read_text())
    if sha(HERE/'records.jsonl')!=report['records_sha256'] or sha(PREVIEW/'manifest.json')!=report['preview_manifest_sha256']:
        raise ValueError('manifest hash mismatch')
    rows=[json.loads(line) for line in (HERE/'records.jsonl').read_text().splitlines()]
    source={r['sample_id']:r for r in json.loads((PREVIEW/'manifest.json').read_text())['records']}
    if len(rows)!=len(source) or len(rows)!=718 or {r['sample_id'] for r in rows}!=set(source):raise ValueError('coverage mismatch')
    counts=collections.Counter();max_error=0.
    for row in rows:
        orig=source[row['sample_id']];path=(ROOT/row['state_path']).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve()) or row['caption']!=orig['caption'] or row['state_sha256']!=orig['state_sha256'] or sha(path)!=row['state_sha256']:
            raise ValueError('state or caption binding mismatch')
        with np.load(path,allow_pickle=False) as f:obj=np.asarray(f['object_wxyz_xyz'],dtype=float);times=np.asarray(f['sample_times'],dtype=float)
        q0=[float(x) for x in obj[0,:4]];n0=math.sqrt(sum(x*x for x in q0));q0=[x/n0 for x in q0]
        p0=[float(x) for x in obj[0,4:]];minz=maxz=p0[2];rise=fall=travel=angle=0.
        for pose in obj:
            q=[float(x) for x in pose[:4]];n=math.sqrt(sum(x*x for x in q));q=[x/n for x in q]
            p=[float(x) for x in pose[4:]];minz=min(minz,p[2]);maxz=max(maxz,p[2])
            rise=max(rise,p[2]-minz);fall=max(fall,maxz-p[2])
            travel=max(travel,math.sqrt(sum((p[i]-p0[i])**2 for i in range(3))))
            angle=max(angle,2*math.acos(min(1.,abs(sum(q[i]*q0[i] for i in range(4))))))
        expected={'ordered_max_rise_m':rise,'ordered_max_fall_m':fall,'max_translation_from_start_m':travel,
                  'max_rotation_from_start_rad':angle,'net_vertical_m':float(obj[-1,6])-p0[2],
                  'end_from_max_height_m':maxz-float(obj[-1,6])}
        for key,value in expected.items():max_error=max(max_error,abs(value-row['metrics'][key]))
        if max_error>1e-7 or row['metrics']['frames']!=len(obj) or row['metrics']['nominal_start_s']!=float(times[0]) or row['metrics']['nominal_end_s']!=float(times[-1]):
            raise ValueError('metric disagreement')
        text=row['caption'].lower();flags=[];pos=report['position_review_threshold_m'];ang=report['angle_review_threshold_rad']
        if 'lift' in text and rise<pos:flags.append('small_observed_center_rise')
        if ('put down' in text or 'set it back down' in text or 'set the largebox back down' in text) and fall<pos:flags.append('small_observed_center_fall')
        if 'move' in text and travel<pos:flags.append('small_observed_center_translation')
        if 'rotate' in text and angle<ang:flags.append('small_observed_object_rotation')
        if any(term in text for term in ('push','pull','kick','release','lift','put down','set it back down')):flags.append('contact_force_and_support_not_observed')
        if flags!=row['review_flags'] or row['training_ready'] is not False:raise ValueError('flag mismatch')
        counts.update(flags)
    if dict(counts)!=report['flag_counts'] or sha(HERE/'records.jsonl')!=report['records_sha256']:
        raise RuntimeError('summary mismatch or input changed')
    result={'source_report_sha256':sha(HERE/'report.json'),'records_sha256':report['records_sha256'],
            'script_sha256':sha(Path(__file__)),'samples':len(rows),'flag_counts':dict(counts),
            'max_scalar_error':max_error,'training_ready':False,
            'limitations':['Uses same processed states, quaternion conventions and engineered triage thresholds.',
                           'Object motion does not establish text, contact, real time, source rights or training eligibility.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
