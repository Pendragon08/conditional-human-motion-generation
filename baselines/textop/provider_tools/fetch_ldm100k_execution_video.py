"""Fetch one already verified small actual clip; partials and client proofs private."""
import argparse
import base64
import hashlib
import json
import os
import pathlib
import shutil
from backup_filter_recovery_checkpoint import recorded_remote, recorded_client
from filter_recovery_windows_paths import guard_local

B = pathlib.Path(__file__).resolve().parents[1]
D = pathlib.Path('D:/具身智能/TextOp复现_20261024')
p = argparse.ArgumentParser()
p.add_argument('--clip', required=True)
a = p.parse_args()
run = 'user_ldm100k_execution_videos_v1'
j = json.loads((B / 'results/ldm100k_user_handoff_20261028' / run / 'progress.json').read_bytes())
matches = [x for x in j['verified_media'] if x['clip'] == a.clip]
assert len(matches) == 1
item = matches[0]
assert item['video_bytes'] < 8 * 1024**2
dest = D / 'media/user100k_effects_20261029' / a.clip
guard_local(D, dest, directory=True)
dest.mkdir(parents=True, exist_ok=True)
video, partial, preview = [dest / x for x in ['video.mp4', 'video.mp4.partial', 'first_frame.png']]
assert not video.exists() and not partial.exists() and not preview.exists(), 'Retain/reconcile any prior client or partial'
assert shutil.disk_usage(D).free > 2 * 1024**3 + item['video_bytes'] + 1024**2
e = dest / 'private_transfer_evidence'
e.mkdir(exist_ok=False)
code = """import pathlib,json,hashlib,base64,os
assert os.getuid()==1013
p=pathlib.Path(VIDEO);rp=p.parent/'receipt.json'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert h(rp)==RECEIPT and h(p)==SHA and p.stat().st_size==BYTES
r=json.loads(rp.read_text());assert r['output_writers_closed'] and r['frames_encoded']==r['frames_decoded'] and r['nvidia_libraries_loaded']==[]
image=p.parent/'first_frame.png';assert h(image)==r['first_frame_sha256'] and image.stat().st_size<1024**2
print(json.dumps(dict(video_sha256=h(p),video_bytes=p.stat().st_size,receipt_sha256=h(rp),first_frame_sha256=h(image),first_frame_base64=base64.b64encode(image.read_bytes()).decode())))
""".replace('VIDEO', repr(item['video_path'])).replace('RECEIPT', repr(item['receipt_sha256'])).replace('SHA', repr(item['video_sha256'])).replace('BYTES', str(item['video_bytes']))
descriptor = json.loads(recorded_remote(code, e / 'source_client.json', timeout=120))
command = ['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30', '-o', 'ServerAliveInterval=10',
           '-i', 'C:/Users/Pende/.ssh/id_ed25519', 'ouyanghao@10.147.17.32:' + item['video_path'], str(partial)]
client = recorded_client('', e / 'copy_client.json', timeout=180, command=command)
assert client['actual_client_exit_code'] == 0 and client['client_reaped'] and client['streams_final']
assert partial.stat().st_size == item['video_bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest() == item['video_sha256']
owner = guard_local(D, partial)
with partial.open('r+b') as f:
    f.flush()
    os.fsync(f.fileno())
partial.replace(video)
raw = base64.b64decode(descriptor.pop('first_frame_base64'))
assert hashlib.sha256(raw).hexdigest() == descriptor['first_frame_sha256']
with preview.open('xb') as f:
    f.write(raw)
    f.flush()
    os.fsync(f.fileno())
guard_local(D, preview)
assert shutil.disk_usage(D).free > 2 * 1024**3
result = dict(status='ACTUAL_LDM100K_EXECUTION_VIDEO_LOCAL_SHA_VERIFIED', clip=a.clip,
    video_path=str(video), first_frame_path=str(preview), **descriptor,
    native_windows_owner=owner, source_provenance=item['provenance'], source_bytes_transferred=item['video_bytes'] + len(raw),
    actual_vpn_billed_bytes_unknown=True, raw_arrays_or_source_annotations_copied=False)
(dest / 'verified.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
(B / 'results/ldm100k_user_handoff_20261028' / run / (a.clip + '_local.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: result[k] for k in ['status', 'video_path', 'first_frame_path', 'video_bytes', 'video_sha256']}))
