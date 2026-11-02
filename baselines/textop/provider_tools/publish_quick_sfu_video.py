"""Publish only the verified SFU noncommercial research-result movie, no motion arrays."""
import hashlib
import argparse
import json
import os
import pathlib
import shutil
import subprocess
from session import remote, GH

B=pathlib.Path(__file__).resolve().parents[1]
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ap=argparse.ArgumentParser();ap.add_argument('--run',default='actual_sample64_longest_completed_v2');a=ap.parse_args()
import re
assert re.fullmatch('[a-z0-9_]+',a.run)
RUN=a.run
REPO='Pendragon08/conditional-human-motion-generation'
TAG='textop-quick-tracking-video-20261026'
license_url='https://download.is.tue.mpg.de/amass/licenses2024/SFU/sfu_license_updated.html'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rec=json.loads((B/'results/quick_pilot_actual_20261026'/RUN/'receipt.json').read_text(encoding='utf-8'))
ex=json.loads((B/'results/quick_pilot_actual_20261026'/RUN/'exit.json').read_text(encoding='utf-8'))
assert rec['status']=='RECORDED_ISAACLAB_POSE_VIDEO_CREATED' and rec['output_writers_closed']
assert ex['supervision_passed'] and ex['child_exit_code']==0
movie=D/'videos/private'/RUN/'video.mp4'
assert sha(movie)==rec['video_sha256']
source=json.loads(remote(r'''
import pathlib,json,hashlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
v=json.loads((r/'recorded_videos_private'/RUN_LITERAL/'receipt.json').read_text())
p=v['provenance'];assert p['pilot_run']=='quick_pilot_sample64_v1'
index=json.loads((r/'tracker_validation'/p['pilot_run']/'executed_traces.private.json').read_text())
entry=index[p['trial_in_finish_order']];assert entry['physical_failure'] is False
m=json.loads((pathlib.Path(entry['folder'])/'trace.private.json').read_text())
view=json.loads((r/'quick_source_views/quick_sample64_nominal_v1/source_decisions.private.json').read_text())
rows=[x for x in view if hashlib.sha256(x['reference']['canonical_source'].encode()).hexdigest()==m['motion_id']]
assert len(rows)==1 and rows[0]['reference']['canonical_source'].split('/')[0]=='SFU'
assert rows[0]['nominal_tracking_completed'] is True
print(json.dumps({'dataset':'SFU','scientific_result_video_only':True,'no_raw_motion_arrays':True}))
'''.replace('RUN_LITERAL',repr(RUN)),timeout=30))
terms=D/'diagnostics/video_license_review_20261026/SFU.html'
assert sha(terms)=='27a0299f51d33d05c17e3a70111c56a555b851431a7d8b2d891495be52e092e8'
public=D/'videos/public_sfu_research_20261026';public.mkdir(parents=True,exist_ok=True)
target=public/'g1_tracker5000_sfu_recorded_execution.mp4'
if target.exists():assert sha(target)==rec['video_sha256']
else:shutil.copy2(movie,target)
assert sha(target)==rec['video_sha256']
notes='''# TextOp quick tracking: recorded G1 execution (SFU research result)

This movie visualizes actual IsaacLab physical states recorded while the G1
privileged tracker follows one SFU reference motion. The tracker checkpoint is
5,000 iterations / 100,000 optimizer updates, not the full 40,000-iteration paper
run. The representative 64-source assessment recorded 42 nominal-duration
completions and 22 failures. This selected clip is the longest executed complete
trial; it is not a text-generator result, full-library success rate or final
deployed WBC demonstration. The last reference endpoint is not independently
checked. CPU/Mesa and MuJoCo are used only for visual FK of recorded states,
with zero additional physics or optimization steps. Visual and tracker assets
are not asserted to have identical collision geometry.

## Motion source and attribution

SFU Motion Capture Database, created by Simon Fraser University and National
University of Singapore: http://mocap.cs.sfu.ca/ . Funding: NUS AcRF grant
R-252-000-429-133 and SFU President’s Research Start-up Grant.

The official SFU terms allow research use, prohibit commercial products/resale,
and request acknowledgment for published results:
https://download.is.tue.mpg.de/amass/licenses2024/SFU/sfu_license_updated.html
This release is a noncommercial scientific result movie. No original AMASS,
body-model parameters, source/text index or retargeted motion arrays are shared.

AMASS processing: Mahmood et al., AMASS (2019), https://amass.is.tue.mpg.de/ .
TextOp source: https://github.com/TeleHuman/Textop .
Robot visual model/retargeting: GMR, https://github.com/YanjieZe/GMR .

Other trials and failures remain in the report and private execution backups.
BMLmovi-derived video is retained privately because its distribution terms are
more restrictive; its numerical failure result remains in aggregate reports.
'''
note=public/'VIDEO_SCOPE_AND_CREDITS.md';note.write_text(notes,encoding='utf-8')
env=dict(os.environ,HTTP_PROXY='http://127.0.0.1:7897',HTTPS_PROXY='http://127.0.0.1:7897',GH_PROMPT_DISABLED='1')
def gh(*args):
    return subprocess.run([GH,*args],capture_output=True,text=True,encoding='utf-8',env=env,timeout=240)
response=gh('api','repos/'+REPO+'/releases/tags/'+TAG)
if response.returncode:
    assert '404' in response.stderr,response.stderr
    made=gh('release','create',TAG,'--repo',REPO,'--target','codex/textop-reproduction-20261024',
        '--title','TextOp 5k quick tracker: actual SFU execution video','--prerelease','--notes-file',str(note))
    assert made.returncode==0,made.stderr
for file in [target,note]:
    response=gh('api','repos/'+REPO+'/releases/tags/'+TAG);assert response.returncode==0,response.stderr
    release=json.loads(response.stdout)
    matches=[x for x in release['assets'] if x['name']==file.name]
    if not matches:
        uploaded=gh('release','upload',TAG,str(file),'--repo',REPO);assert uploaded.returncode==0,uploaded.stderr
    response=gh('api','repos/'+REPO+'/releases/tags/'+TAG);assert response.returncode==0,response.stderr
    asset=next(x for x in json.loads(response.stdout)['assets'] if x['name']==file.name)
    assert asset['size']==file.stat().st_size and asset['digest']=='sha256:'+sha(file)
    if file==target:movie_asset=asset
proof={'status':'SFU_RESEARCH_RESULT_VIDEO_GITHUB_SHA_VERIFIED','run':RUN,
       'video_sha256':rec['video_sha256'],'github_release':release['html_url'],
       'github_video':movie_asset['browser_download_url'],'github_digest':movie_asset['digest'],
       'motion_dataset':'SFU','license_url':license_url,'license_snapshot_sha256':sha(terms),
       'scope':'noncommercial research result video only; not raw motion redistribution',
       'bmlmovi_video_kept_private':True,'raw_motion_or_text_index_published':False,
       'full_paper_reproduction_claimed':False}
folder=B/'results/quick_pilot_actual_20261026'/RUN
(folder/'github_video.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
(D/'GitHub视频同步记录_SFU_20261026.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof))
