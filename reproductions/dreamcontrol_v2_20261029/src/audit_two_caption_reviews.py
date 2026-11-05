"""Verify selected original captions, receipt binding and every displayed frame."""
import hashlib
import json
from pathlib import Path
import zipfile
import cv2
import numpy as np

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
BASE=ROOT/'six_family_disjoint_preview_v2_20261101'
SHEETS=ROOT/'five_all_frames_review_20261101'
VIDEOS=ROOT/'five_caption_review_videos_20261031'
ZIP=Path('/data/code/ouyanghao/oyh/phygile-reproduction-20261017/data/amass/humanml_pairing_20261021/code/public/HumanML3D/texts.zip')
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    sheet=json.loads((SHEETS/'report.json').read_text())
    records={r['sample_id']:r for r in json.loads((BASE/'manifest.json').read_text())['records']}
    states={r['sample_id']:r for r in json.loads((BASE/'state_manifest.json').read_text())['records']}
    summary=[]
    for sid,clip in [('000008','000008'),('ACCAD_004347','004347')]:
        receipt=HERE/(sid+'_review.json'); r=json.loads(receipt.read_text()); rec=records[sid];state=states[sid]
        assert r['sample_id']==sid and r['candidate_sha256']==sha(BASE/rec['feature_path']) and r['state_sha256']==sha(BASE/state['state_path'])
        assert r['all_frame_sheet_report_sha256']==sha(SHEETS/'report.json')
        with zipfile.ZipFile(ZIP) as z:raw=z.read('texts/'+clip+'.txt')
        parts=raw.decode('utf8').splitlines()[r['selected_line']].split('#')
        assert hashlib.sha256(raw).hexdigest()==r['caption_member_sha256'] and sha(ZIP)==r['caption_archive_sha256']
        assert parts[0]==r['caption'] and parts[2:4]==r['tags']
        with np.load(BASE/state['state_path'],allow_pickle=False) as f:t=f['sample_times'].copy()
        np.testing.assert_allclose(r['sample_range_s'],[t[0],t[-1]],atol=1e-12,rtol=0)
        assert r['source_range_s'][0]<=t[0]<t[-1]<=r['source_range_s'][1]
        case=next(x for x in sheet['cases'] if x['sample_id']==sid)
        assert sha(VIDEOS/(sid+'.mp4'))==r['source_video_sha256']==case['video_sha256']
        capture=cv2.VideoCapture(str(VIDEOS/(sid+'.mp4'))); index=0
        for page in case['pages']:
            path=SHEETS/page['path']; assert sha(path)==page['sha256']
            img=cv2.imread(str(path));assert img.shape==(1200,2000,3)
            for i in range(page['first_frame'],page['last_frame']+1):
                assert i==index
                ok,frame=capture.read();assert ok
                frame=cv2.resize(frame,(400,300),interpolation=cv2.INTER_AREA)
                cv2.putText(frame,f'{sid} frame {index:03d}',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
                k=index-page['first_frame']; y=(k//5)*300;x=(k%5)*400
                assert np.array_equal(frame,img[y:y+300,x:x+400]);index+=1
        ok,_=capture.read();capture.release();assert not ok and index==case['frames']==len(t)
        out={'receipt_sha256':sha(receipt),'source_video_sha256':r['source_video_sha256'],
             'frames_pixel_checked':index,'training_ready':False,'script_sha256':sha(Path(__file__)),
             'limitations':'Same decoded videos and renderer; no independent semantic reviewer, true source or physical acceptance.'}
        with (HERE/(sid+'_audit.json')).open('x') as f:json.dump(out,f,indent=2);f.write('\n')
        summary.append({'sample_id':sid,'receipt_sha256':out['receipt_sha256'],'audit_sha256':sha(HERE/(sid+'_audit.json')),'frames_checked':index})
    with (HERE/'independent_audit.json').open('x') as f:json.dump({'cases':summary,'training_ready':False},f,indent=2);f.write('\n')
    print(json.dumps({'cases':len(summary),'frames_checked':sum(x['frames_checked'] for x in summary),'training_ready':False}))


if __name__=='__main__':main()
