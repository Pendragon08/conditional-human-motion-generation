"""Recompute the full 1,514-variant text and nominal time catalog."""
import collections
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent
ARCHIVE=ROOT/'data/OmniRetarget_Dataset/robot-object.zip'
ANNOT=ROOT/'omomo_text_pairing_20261101/omomo_text_anno.zip'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def distance(a,b):return math.sqrt(sum((float(a[k])-float(b[k]))**2 for k in range(3)))


def main():
    report=json.loads((HERE/'report.json').read_text())
    if sha(HERE/'records.jsonl')!=report['records_sha256'] or sha(ARCHIVE)!=report['archive_sha256'] or sha(ANNOT)!=report['annotation_archive_sha256']:
        raise ValueError('source hashes differ')
    rows=[json.loads(line) for line in (HERE/'records.jsonl').read_text().splitlines()]
    if len(rows)!=1514 or len({r['motion_member'] for r in rows})!=1514:raise ValueError('coverage differs')
    captions=collections.Counter();cropped=0;errors=[];diffs=[]
    with zipfile.ZipFile(ARCHIVE) as archive, zipfile.ZipFile(ANNOT) as annotations:
        for row in rows:
            raw=archive.read(row['motion_member']);text=annotations.read(row['annotation_member'])
            if hashlib.sha256(raw).hexdigest()!=row['motion_member_sha256'] or hashlib.sha256(text).hexdigest()!=row['annotation_sha256']:
                raise ValueError('member hashes differ')
            if json.loads(text)[row['nominal_action_key']]!=row['caption']:raise ValueError('caption differs')
            with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                q=f['qpos'];fps=float(f['fps'])
            if fps!=30:raise ValueError('unexpected nominal fps')
            n=min(len(q),300);last=math.floor(((n-1)/fps)*20+1e-9);pos=last*1.5;lo=math.floor(pos);frac=pos-lo
            end=[float(q[lo,40+k])*(1-frac)+float(q[lo+int(frac>0),40+k])*frac for k in range(3)]
            final=distance(q[-1,40:43],end)
            later=q[lo+1:,40:43] if frac else q[lo:,40:43]
            maximum=max((distance(p,end) for p in later),default=0.)
            t=row['timeline']
            if (t['source_frames']!=len(q) or t['crop_frames']!=n or t['source_tail_frames_after_crop']!=len(q)-n or
                t['nominal_feature_rows']!=last or t['target_last_source_frame_position']!=pos or t['full_caption_valid_for_variant_or_window'] is not False):
                raise ValueError('time mapping differs')
            errors.extend((abs(final-t['target_to_full_end_center_m']),abs(maximum-t['target_to_later_max_center_m'])))
            if max(errors[-2:])>1e-10:raise ValueError('object endpoint differs')
            captions[row['caption']]+=1;cropped+=int(len(q)>n);diffs.append(final)
    if dict(captions)!=report['caption_distribution'] or cropped!=report['cropped_before_source_end']:
        raise ValueError('summary differs')
    if not np.allclose(np.quantile(diffs,[0,.5,.9,1]),report['target_to_full_end_center_m_quantiles'],rtol=0,atol=1e-10):
        raise ValueError('quantiles differ')
    if sha(ARCHIVE)!=report['archive_sha256'] or sha(HERE/'records.jsonl')!=report['records_sha256']:
        raise RuntimeError('input changed')
    result={'source_report_sha256':sha(HERE/'report.json'),'records_sha256':report['records_sha256'],
            'script_sha256':sha(Path(__file__)),'variants':len(rows),'cropped_before_source_end':cropped,
            'max_scalar_coordinate_error_m':max(errors),'caption_distribution':dict(captions),'training_ready':False,
            'limitations':['Independent scalar formulas use the same raw trajectories and source text archive.',
                           'Nominal time and linear object interpolation only; no semantic, source license, real clock or physical acceptance.']}
    with (HERE/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
