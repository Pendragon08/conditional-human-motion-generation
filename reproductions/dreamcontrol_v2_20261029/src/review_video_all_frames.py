"""Make all-frame contact sheets from previously audited comparison videos."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
VIDEOS=ROOT/'five_caption_review_videos_20261031'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior=json.loads((VIDEOS/'report.json').read_text()); audit=json.loads((VIDEOS/'audit_report.json').read_text())
    assert sha(VIDEOS/'report.json')==audit['report_sha256']
    cases={r['sample_id']:r for r in prior['cases']}; results=[]
    for sid in ('000008','003017','ACCAD_004347'):
        case=cases[sid]; video=VIDEOS/(sid+'.mp4'); assert sha(video)==case['video_sha256']
        capture=cv2.VideoCapture(str(video))
        if not capture.isOpened():raise ValueError('video cannot be opened')
        fps=capture.get(cv2.CAP_PROP_FPS)
        if abs(fps-20)>1e-6:raise ValueError('unexpected video fps')
        frames=[]
        while True:
            ok,frame=capture.read()
            if not ok:break
            frame=cv2.resize(frame,(400,300),interpolation=cv2.INTER_AREA)
            cv2.putText(frame,f'{sid} frame {len(frames):03d}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
            frames.append(frame)
        capture.release()
        if len(frames)!=case['frames']:raise ValueError('decoded frame count differs from prior record')
        pages=[]
        for start in range(0,len(frames),20):
            grid=np.full((1200,2000,3),255,dtype=np.uint8)
            for j,frame in enumerate(frames[start:start+20]):
                y=(j//5)*300;x=(j%5)*400;grid[y:y+300,x:x+400]=frame
            path=HERE/f'{sid}_{start:03d}.png'
            if path.exists():raise FileExistsError(path)
            if not cv2.imwrite(str(path),grid):raise IOError('could not save sheet')
            pages.append({'path':path.name,'sha256':sha(path),'first_frame':start,'last_frame':min(start+19,len(frames)-1)})
        results.append({'sample_id':sid,'video_sha256':sha(video),'frames':len(frames),'pages':pages,
                        'feature_sha256':case['feature_sha256'],'captions':case['caption_records'],
                        'caption_accepted':False})
    report={'cases':results,'video_report_sha256':sha(VIDEOS/'report.json'),
            'script_sha256':sha(Path(__file__)),'training_ready':False,
            'limitations':'Review material from previously rendered points; no new source, model, timing or physics validation.'}
    with (HERE/'report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'cases':len(results),'total_frames':sum(x['frames'] for x in results),'training_ready':False}))


if __name__=='__main__':main()
