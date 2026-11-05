"""Extract every frame of the previously audited MoSh comparison video."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
SOURCE=ROOT/'mosh_human_robot_review_20261031'
HERE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior=json.loads((SOURCE/'report.json').read_text());audit=json.loads((SOURCE/'audit_report.json').read_text())
    assert sha(SOURCE/'report.json')==audit['report_sha256']
    video=SOURCE/'human_robot_review.mp4';assert sha(video)==prior['video_sha256']
    cap=cv2.VideoCapture(str(video));assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-20)<1e-6
    frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frame=cv2.resize(frame,(400,300),interpolation=cv2.INTER_AREA)
        cv2.putText(frame,f'frame {len(frames):03d}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
        frames.append(frame)
    cap.release();assert len(frames)==prior['frames']==149
    pages=[]
    for start in range(0,len(frames),20):
        grid=np.full((1200,2000,3),255,dtype=np.uint8)
        for k,frame in enumerate(frames[start:start+20]):grid[(k//5)*300:(k//5+1)*300,(k%5)*400:(k%5+1)*400]=frame
        path=HERE/f'000798_{start:03d}.png'
        if path.exists():raise FileExistsError(path)
        if not cv2.imwrite(str(path),grid):raise IOError('failed to write')
        pages.append({'path':path.name,'sha256':sha(path),'first_frame':start,'last_frame':min(start+19,len(frames)-1)})
    out={'sample_id':'MOSH_000798_crop','frames':len(frames),'video_sha256':sha(video),'pages':pages,
         'candidate_sha256':prior['candidate_sha256'],'source_report_sha256':sha(SOURCE/'report.json'),
         'original_caption_records':prior['original_caption_records'],'training_ready':False,
         'script_sha256':sha(Path(__file__)),
         'limitations':'Review material from approximate motion; source, actual time, ground and physics remain unaccepted.'}
    with (HERE/'report.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({'frames':len(frames),'pages':len(pages),'training_ready':False}))


if __name__=='__main__':main()
