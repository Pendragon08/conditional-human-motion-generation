"""Serial CPU/Mesa media from verified captured states; no control or training."""
import argparse
import fcntl
import json
import os
import pathlib
import shutil
import time
from contact_process_supervision import run_supervised
from contact_filter_resources import sample_resources
from ldm100k_verified_input import ROOT, owned, read, sha

p = argparse.ArgumentParser()
p.add_argument('--request', required=True)
a = p.parse_args()
sp = owned(a.request)
s = read(sp)
assert os.getuid() == 1013 and sha(__file__) == s['supervisor_sha256']
assert sha(ROOT / 'scripts/render_ldm100k_execution_video.py') == s['renderer_sha256']
assert sha(ROOT / 'configs/execution-video-spec.json') == s['prior_visual_spec_sha256']
lock = (ROOT / 'receipts/full_library_capacity.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
out = owned(s['output_root'], pathlib.Path('/home/ouyanghao'))
assert not (out / 'batch_receipt.json').exists()
assert shutil.disk_usage(out).free > 2 * 1024**3 + 128 * 1024**2 + 64 * 1024**2
rec = dict(status='RUNNING_LDM100K_RECORDED_POSE_VIDEOS', run=s['run'], request_sha256=sha(sp),
    source_sha256=sha(__file__), declared_clips=len(s['selected_trial_indices']), completed_clips=[],
    selection='first lexicographic segment/seed pair, plus every physical failure; deduplicated',
    new_neural_model_loads=0, new_sdk_tasks=0, new_physics_steps=0, optimizer_updates=0,
    output_writers_closed=False)
start = time.monotonic()


def save():
    rec['elapsed_seconds'] = time.monotonic() - start
    temp = out / 'batch_receipt.partial'
    with temp.open('w') as f:
        json.dump(rec, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(out / 'batch_receipt.json')


def sampler(pid):
    v = sample_resources(ROOT, pid, 2 * 1024**3, 2 * 1024**3 + 128 * 1024**2)
    v['home_free_bytes'] = shutil.disk_usage(out).free
    assert v['home_free_bytes'] > 2 * 1024**3 + 128 * 1024**2
    return v


save()
for trial in s['selected_trial_indices']:
    clip = 'ldm100k_trial_' + str(trial)
    cmd = ['taskset', '-c', '0-3', '/usr/bin/env'] + [k + '=' + v for k, v in s['environment'].items()] + [
        s['python'], str(ROOT / 'scripts/render_ldm100k_execution_video.py'), '--spec', str(sp), '--mode', 'actual',
        '--run', clip, '--pilot-run', 'user_ldm100k_generated_physics_v1', '--trial', str(trial)]
    proof = run_supervised(cmd, out / (clip + '.log'), out / (clip + '_resources.jsonl'),
        out / (clip + '_process_exit.json'), sampler, 300, 15)
    assert proof['supervision_passed']
    rp = out / clip / 'receipt.json'
    r = read(rp)
    assert r['status'] == 'RECORDED_ISAACLAB_POSE_VIDEO_CREATED' and r['output_writers_closed']
    assert r['new_physics_steps'] == r['new_optimizer_updates'] == r['sdk_runs'] == 0
    assert r['nvidia_libraries_loaded'] == [] and 'llvmpipe' in r['gl_renderer'].lower()
    assert r['frames_encoded'] == r['frames_decoded'] and r['fps'] == 50
    video = out / clip / 'video.mp4'
    assert sha(video) == r['video_sha256'] and video.stat().st_size == r['video_bytes']
    rec['completed_clips'].append(dict(clip=clip, trial_in_finish_order=trial, receipt_sha256=sha(rp),
        video_path=str(video), video_sha256=sha(video), video_bytes=video.stat().st_size,
        frames=r['frames_decoded'], provenance=r['provenance']))
    assert sum(x['video_bytes'] for x in rec['completed_clips']) <= 64 * 1024**2
    save()
rec.update(status='USER_LDM100K_RECORDED_EXECUTION_VIDEOS_FINISHED', output_writers_closed=True)
save()
print(json.dumps(rec), flush=True)
