"""Record manually observed content and bind every reviewed sheet by hash."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pair=R/'reports/kit_dart_pair_review_audit_20261102.json';source=R/'reports/kit_dart_robot_export_audit_20261102.json';folder=R/'data/kit_dart_pair_review_20261102'
p=json.loads(pair.read_text());s=json.loads(source.read_text());a=next(x for x in p['cases'] if x['id']=='006923');b=next(x for x in s['cases'] if x['id']=='006923')
if a['aligned_frames']!=153 or a['sheets']!=20 or a['robot_sha256']!=b['preview_sha256']:raise ValueError('binding mismatch')
files=[folder/f'006923_{i:03d}.png' for i in range(0,153,8)]
if len(files)!=20 or not all(x.is_file() for x in files):raise ValueError('missing sheets')
out={'id':'006923','reviewed_frames':153,'sheets':[{'file':f.name,'sha256':sha(f)} for f in files],
 'pair_audit_sha256':sha(pair),'render_report_sha256':p['render_report_sha256'],'human_sha256':a['human_sha256'],'robot_sha256':a['robot_sha256'],'caption_sha256':b['caption_sha256'],
 'observations':['The person is nearly stationary in the first long portion; a leg then swings and steps while the body shifts, followed by an approximately stationary stance.', 'The G1 preview preserves a broadly similar stepping event in the two projections.'],
 'unresolved':['The two projections do not establish whether the step is backward relative to facing direction, nor an external push, foot contact, or dynamic feasibility.', 'Source identity and actual clock are unconfirmed; display uses a 100 Hz hypothesis and misses the first human sample.'],
 'caption_accepted':False,'training_ready':False,'review_method':'manual examination of all 20 contact sheets and 153 aligned sampled frames'}
with (R/'reports/kit_dart_006923_visual_review_20261103.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps({'frames':out['reviewed_frames'],'sheets':len(files),'caption_accepted':False}))
