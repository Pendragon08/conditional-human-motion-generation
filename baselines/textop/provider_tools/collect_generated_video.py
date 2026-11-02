"""Verify a closed recorded-pose movie and keep the media private on D:."""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
from session import remote

B=pathlib.Path(__file__).resolve().parents[1]
D=pathlib.Path('D:/具身智能/TextOp复现_20261024')
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
assert re.fullmatch('[a-z0-9_]+',a.run)
script=r'''
import pathlib,json,hashlib
r=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024');run=RUN_LITERAL
folder=r/'recorded_videos_private'/run;rp=folder/'receipt.json'
ep=r/'receipts'/('generated_video_'+run+'_exit.json')
assert ep.exists(), 'No final exit proof yet: do not collect an active renderer'
rec=json.loads(rp.read_text());ex=json.loads(ep.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert rec['status']=='RECORDED_ISAACLAB_POSE_VIDEO_CREATED' and rec['output_writers_closed'] is True
assert rec['mode']=='actual' and rec['actual_policy_result_video'] is True
assert rec['provenance']['source_kind']=='trained_generated_reference_actual_isaaclab_execution'
assert rec['provenance']['wrapper_sha256']==sha(r/'scripts/render_generated_execution.py')
assert rec['provenance']['render_spec_sha256']==sha(r/'configs/generated-execution-video-spec.json')
assert rec['frames_encoded']==rec['frames_decoded'] and rec['fps']==50
assert rec['new_physics_steps']==rec['new_optimizer_updates']==rec['sdk_runs']==0
assert rec['nvidia_libraries_loaded']==[] and 'llvmpipe' in rec['gl_renderer'].lower()
assert ex['supervision_passed'] is True and ex['child_exit_code']==0
assert ex['child_reaped'] is True and ex['timed_out'] is False and ex['supervisor_error'] is None
assert ex['evaluation_receipt_sha256']==sha(rp)
for p,k in [(r/'receipts'/('generated_video_'+run+'_process_exit.json'),'process_exit_sha256'),
            (r/'logs'/('generated_video_'+run+'_resources.jsonl'),'resources_sha256')]:assert sha(p)==ex[k]
video=folder/'video.mp4';frame=folder/'first_frame.png'
assert sha(video)==rec['video_sha256'] and sha(frame)==rec['first_frame_sha256']
assert video.stat().st_size==rec['video_bytes']
print(json.dumps({'receipt':rec,'exit':ex,'media':[dict(path=str(p),name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in [video,frame]]}))
'''.replace('RUN_LITERAL',repr(a.run))
j=json.loads(remote(script,timeout=40))
private=D/'videos/private'/a.run;private.mkdir(parents=True,exist_ok=True)
for item in j['media']:
    target=private/item['name']
    if not target.exists():
        assert shutil.disk_usage(private).free>item['bytes']+1024**3
        partial=target.with_suffix(target.suffix+'.partial')
        assert not partial.exists(), 'Interrupted transfer preserved; inspect before resume'
        p=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',
            'C:/Users/Pende/.ssh/id_ed25519','ouyanghao@10.147.17.32:'+item['path'],str(partial)],
            capture_output=True,text=True,encoding='utf-8',timeout=120)
        assert p.returncode==0,p.stderr
        assert partial.stat().st_size==item['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==item['sha256']
        partial.replace(target)
    assert target.stat().st_size==item['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
public=B/'results/generated_execution_20261026'/a.run;public.mkdir(parents=True,exist_ok=True)
local=D/'results/generated_execution_20261026'/a.run;local.mkdir(parents=True,exist_ok=True)
for key in ['receipt','exit']:
    data=json.dumps(j[key],indent=2).encode('utf-8')
    for parent in [private,public,local]:(parent/(key+'.json')).write_bytes(data)
proof={'status':j['receipt']['status'],'frames':j['receipt']['frames_decoded'],
       'video_sha256':j['receipt']['video_sha256'],'private_server_and_local_sha_verified':True,
       'private_media_publicly_published':False,'new_physics_steps':0,'sdk_runs':0}
(public/'media_backup.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof))
