"""Independently check MoSh caption provenance, review frames, and wrist statistics."""
import hashlib
import json
from pathlib import Path
import zipfile
import cv2
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
HUMAN=ROOT/'mosh_000798_fullrate_20261031/output'
CROP=ROOT/'mosh_000798_crop_candidate_20261031'
REVIEW=ROOT/'mosh_all_frames_review_20261101'
VIDEO=ROOT/'mosh_human_robot_review_20261031/human_robot_review.mp4'
ZIP=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    receipt=HERE/'semantic_review.json';r=json.loads(receipt.read_text())
    view=json.loads((REVIEW/'report.json').read_text());movement=json.loads((REVIEW/'wrist_motion.json').read_text())
    assert sha(REVIEW/'report.json')==r['all_frame_review_sha256'] and sha(REVIEW/'wrist_motion.json')==r['wrist_measurements_sha256']
    assert sha(VIDEO)==r['source_video_sha256']==view['video_sha256'] and sha(ZIP)==r['caption_archive_sha256']
    with zipfile.ZipFile(ZIP) as z:raw=z.read('texts/000798.txt')
    parts=raw.decode('utf8').splitlines()[r['selected_line']].split('#')
    assert hashlib.sha256(raw).hexdigest()==r['caption_member_sha256'] and parts[0]==r['caption'] and parts[2:4]==r['tags']
    rec=next(x for x in json.loads((BASE/'manifest.json').read_text())['records'] if x['sample_id']==r['sample_id'])
    state=next(x for x in json.loads((BASE/'state_manifest.json').read_text())['records'] if x['sample_id']==r['sample_id'])
    assert sha(BASE/rec['feature_path'])==r['candidate_sha256'] and sha(BASE/state['state_path'])==r['state_sha256']
    with np.load(BASE/state['state_path'],allow_pickle=False) as d:t=d['sample_times'].copy()
    np.testing.assert_allclose(r['sample_range_s'],[t[0],t[-1]],atol=1e-12,rtol=0)
    assert r['source_range_s'][0]<=t[0]<t[-1]<=r['source_range_s'][1]
    np.testing.assert_allclose(r['excluded_head_tail_s'],[t[0]-r['source_range_s'][0],r['source_range_s'][1]-t[-1]],atol=1e-12,rtol=0)
    cap=cv2.VideoCapture(str(VIDEO));index=0
    for page in view['pages']:
        path=REVIEW/page['path'];assert sha(path)==page['sha256'];sheet=cv2.imread(str(path));assert sheet.shape==(1200,2000,3)
        for i in range(page['first_frame'],page['last_frame']+1):
            assert i==index
            ok,frame=cap.read();assert ok
            frame=cv2.resize(frame,(400,300),interpolation=cv2.INTER_AREA)
            cv2.putText(frame,f'frame {index:03d}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
            k=index-page['first_frame'];y=(k//5)*300;x=(k%5)*400
            assert np.array_equal(frame,sheet[y:y+300,x:x+400]);index+=1
    ok,_=cap.read();cap.release();assert not ok and index==149==len(t)
    with np.load(CROP/'candidate.npz',allow_pickle=False) as d:bot=d['points_z_up_world'].astype(float);frames=d['frames_20hz'].copy()
    with np.load(HUMAN/'human_120hz.npz',allow_pickle=False) as d:raw_h=d['human'].astype(float);source_frames=d['source_frames'].copy()
    pos=np.searchsorted(source_frames,frames);assert np.array_equal(source_frames[pos],frames)
    for kind,points in [('human',raw_h[pos]),('robot',bot)]:
        for side,s,w in [('left',16,20),('right',17,21)]:
            relative=[[float(points[i,w,k]-points[i,s,k]) for k in range(3)] for i in range(len(points))]
            ranges=[max(row[k] for row in relative)-min(row[k] for row in relative) for k in range(3)]
            dist=sum(sum((relative[i][k]-relative[i-1][k])**2 for k in range(3))**.5 for i in range(1,len(relative)))
            np.testing.assert_allclose(ranges,movement[kind][side]['coordinate_ranges_m'],rtol=0,atol=1e-6)
            assert abs(dist-movement[kind][side]['cumulative_displacement_m'])<1e-5
    out={'receipt_sha256':sha(receipt),'frames_pixel_checked':index,'caption_member_sha256':hashlib.sha256(raw).hexdigest(),
         'wrist_statistics_recomputed':True,'training_ready':False,'script_sha256':sha(Path(__file__)),
         'limitations':'Same source and video rendering; no independent reviewer, finger motion, ground or physical acceptance.'}
    with (HERE/'independent_audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))


if __name__=='__main__':main()
