"""Audit 15 original human video files against logged hashes and nominal robot timestamps."""
import hashlib,json
from pathlib import Path
import cv2
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
D=R/'omomo_longclip_source_video_v2_20261103';U=R/'omni29_uncropped_review_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 events=[json.loads(x) for x in (D/'events.jsonl').read_text().splitlines()]
 completed=[x for x in events if x['event']=='complete'];found={x['sample']:x for x in events if x['event']=='extracted'}
 if len(completed)!=1 or completed[0]['found']!=15 or len(found)!=15:raise ValueError('incomplete extraction')
 ur=json.loads((U/'report.json').read_text());rec=U/'records.jsonl'
 if sha(rec)!=ur['records_sha256']:raise ValueError('reference hash')
 rows=[json.loads(x) for x in rec.read_text().splitlines()]
 originals={x['nominal_source_group'].split(':',1)[1]:x for x in rows if x['source_member'].endswith('_original.npz')}
 if set(originals)!=set(found):raise ValueError('nominal group coverage')
 details=[]
 for name,e in sorted(found.items()):
  p=D/(name+'.mp4')
  if sha(p)!=e['sha256'] or p.stat().st_size!=e['bytes']:raise ValueError('video checksum')
  cap=cv2.VideoCapture(str(p));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  if not cap.isOpened() or frames<1 or fps<=0:raise ValueError('video metadata')
  decoded=0
  while True:
   ok,_=cap.read()
   if not ok:break
   decoded+=1
  cap.release()
  r=originals[name]
  if decoded!=frames:raise ValueError('video decode count')
  details.append({'group':name,'video_sha256':e['sha256'],'video_frames':frames,'decoded_frames':decoded,'video_fps':fps,
   'width':w,'height':h,'robot_source_frames':r['source_frames'],'robot_source_fps':r['source_fps_nominal'],
   'robot_source_member_sha256':r['source_sha256'],'nominal_length_difference_s':(frames-1)/fps-(r['source_frames']-1)/r['source_fps_nominal']})
 out={'videos':len(details),'extraction_event_sha256':sha(D/'events.jsonl'),'robot_records_sha256':sha(rec),
      'details':details,'max_nominal_length_difference_s':max(abs(x['nominal_length_difference_s']) for x in details),
      'caption_accepted':0,'training_ready':False,'limitations':['Name and metadata agreement do not prove independent recording identity or clock synchronization.',
      'Video decodability does not prove caption semantics or legal scope.']}
 with (D/'audit.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps({k:v for k,v in out.items() if k!='details'}))
if __name__=='__main__':main()
