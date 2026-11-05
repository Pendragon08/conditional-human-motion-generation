"""Audit videos and materialize every decoded frame into indexed review pages.

This checks source bindings and playback coverage, not render truth or caption
semantics. No object mesh, source-frame physics, or contact is established.
"""
import collections
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
PREVIEW=ROOT/'omomo_processed_full_preview_20261101'
VIDEOS=ROOT/'omomo_motion_review_20261101'
HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    report=json.loads((VIDEOS/'report.json').read_text())
    manifest=PREVIEW/'manifest.json'
    if sha(manifest)!=report['preview_manifest_sha256'] or sha(VIDEOS/'render_omomo_motion_review.py')!=report['script_sha256']:
        raise ValueError('input hash mismatch')
    candidates=json.loads(manifest.read_text())['records']; groups=collections.defaultdict(list)
    for row in candidates:
        if row['source_member'].endswith('_original.npz') and row['processing_evidence_folder']!='omni_long15_gpu_20261101' and row['feature_frames']>=40:
            groups[row['caption']].append(row)
    chosen={}
    for caption,rows in groups.items():
        rows.sort(key=lambda r:(r['feature_frames'],r['sample_id']))
        chosen[caption]=rows[len(rows)//2]
    if len(chosen)!=7 or len(report['cases'])!=7:raise ValueError('caption coverage mismatch')
    records=[]
    for case in report['cases']:
        source=chosen[case['caption']]; path=(ROOT/source['state_path']).resolve(strict=True)
        if (case['sample_id']!=source['sample_id'] or case['state_sha256']!=source['state_sha256'] or
            case['source_sha256']!=source['source_sha256'] or case['split']!=source['auxiliary_split'] or
            case['caption_accepted'] is not False or not path.is_relative_to(ROOT.resolve()) or sha(path)!=case['state_sha256']):
            raise ValueError('case source binding mismatch')
        video=VIDEOS/case['video']
        if video.name!=case['sample_id']+'.mp4' or sha(video)!=case['video_sha256']:
            raise ValueError('video hash mismatch')
        with np.load(path,allow_pickle=False) as f:
            n=len(f['sample_times'])
            if n!=case['frames'] or n!=source['feature_frames']+1:raise ValueError('frame count mismatch')
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-20)>1e-6:raise ValueError('video cannot be decoded at nominal FPS')
        pages=[]; frames=[]; count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            frame=cv2.resize(frame,(540,180),interpolation=cv2.INTER_AREA)
            cv2.putText(frame,f'{case["sample_id"]} frame {count:03d}',(5,12),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,0,0),1,cv2.LINE_AA)
            frames.append(frame);count+=1
            if len(frames)==20 or count==n:
                if shutil.disk_usage(HERE).free < 8*1024**3:raise RuntimeError('disk guard')
                canvas=np.full((720,2700,3),255,dtype=np.uint8)
                for i,image in enumerate(frames):canvas[(i//5)*180:(i//5+1)*180,(i%5)*540:(i%5+1)*540]=image
                page=HERE/f'{case["sample_id"]}_{count-len(frames):03d}.png'
                if page.exists() or not cv2.imwrite(str(page),canvas):raise IOError('cannot create page')
                pages.append({'file':page.name,'sha256':sha(page),'first':count-len(frames),'last':count-1})
                frames=[]
        cap.release()
        if count!=n or frames:raise ValueError('decoded video coverage mismatch')
        records.append({'sample_id':case['sample_id'],'caption':case['caption'],'frames':count,
                        'video_sha256':case['video_sha256'],'state_sha256':case['state_sha256'],'pages':pages})
    if sha(manifest)!=report['preview_manifest_sha256']:raise RuntimeError('manifest changed')
    result={'render_report_sha256':sha(VIDEOS/'report.json'),'script_sha256':sha(__file__),'cases':records,
            'total_frames':sum(x['frames'] for x in records),'training_ready':False,
            'limitations':['Frame count and hashes do not independently validate rendered coordinates.',
                           'All-frame pages still omit object shape, surfaces and physical contact.',
                           'No caption has been accepted by materializing these pages.']}
    with (HERE/'audit.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'cases':len(records),'frames':result['total_frames'],'pages':sum(len(x['pages']) for x in records)}))


if __name__=='__main__':main()
