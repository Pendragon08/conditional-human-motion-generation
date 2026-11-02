"""Serial, exclusively locked CPU/Mesa rendering of recorded 1000-policy poses."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import time
import traceback

ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
RUN='user_mixed1000_execution_videos_v1'
FLOOR=2*1024**3+128*1024**2+128*1024**2+64*1024**2

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--trusted-request-sha256',required=True);a=p.parse_args()
    request=pathlib.Path(a.request)
    if not __debug__ or os.getuid()!=1013 or request.parent!=ROOT/'receipts' or request.resolve()!=request or request.stat().st_uid!=1013 or sha(request)!=a.trusted_request_sha256:
        raise ValueError('invalid request identity')
    job=json.loads(request.read_text())
    if job['run']!=RUN or job['authorized'] is not True or sha(__file__)!=job['supervisor_sha256']:
        raise ValueError('wrong video request')
    for name,digest in job['bound_sources'].items():
        path=ROOT/name
        if pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts or path.resolve()!=path or path.stat().st_uid!=1013 or sha(path)!=digest:
            raise ValueError('source differs: '+name)
    from contact_process_supervision import run_supervised
    from contact_filter_resources import sample_resources
    os.sched_setaffinity(0,{0,1,2,3})
    output=ROOT/'step34_20261029'/RUN
    if job['output_root']!=str(output) or len(job['selected_trial_indices'])!=3:raise ValueError('invalid media scope')
    start=time.monotonic();rec=dict(status='STARTING_MIXED1000_EXECUTION_POSE_VIDEOS',run=RUN,
        request_sha256=sha(request),declared_clips=3,completed_clips=[],
        selection='longest completed conditional traces from distinct segments; deterministic ties',
        new_model_loads=0,new_sdk_tasks=0,new_physics_steps=0,optimizer_updates=0,output_writers_closed=False)
    with (ROOT/'receipts/full_library_capacity.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if output.exists() or shutil.disk_usage(ROOT).free<FLOOR:raise RuntimeError('existing output or insufficient space')
        for item in job['physics_binding'].values():
            path=pathlib.Path(item['path'])
            if path.resolve()!=path or path.stat().st_uid!=1013 or sha(path)!=item['sha256']:raise ValueError('physics closure changed')
        sample_resources(ROOT,None,2*1024**3,FLOOR)
        output.mkdir(mode=0o700,exist_ok=False)
        def save():
            rec['elapsed_seconds']=time.monotonic()-start
            temp=output/'batch_receipt.partial'
            with temp.open('w') as f:json.dump(rec,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
            temp.replace(output/'batch_receipt.json')
        def sample(pid):
            if shutil.disk_usage(ROOT).free<FLOOR:raise RuntimeError('disk reserve unavailable')
            return sample_resources(ROOT,pid,2*1024**3,FLOOR)
        save()
        try:
            for trial in job['selected_trial_indices']:
                clip='mixed1000_trial_'+str(trial)
                cmd=['taskset','-c','0-3','/usr/bin/env','-u','PYTHONPATH','-u','PYTHONOPTIMIZE']
                cmd += [k+'='+v for k,v in job['environment'].items()]
                cmd += [job['python'],str(ROOT/'scripts/render_mixed1000_execution_video_v1.py'),
                        '--spec',str(request),'--trusted-request-sha256',a.trusted_request_sha256,
                        '--mode','actual','--run',clip,'--pilot-run','user_augmentation2526_mixed1000_heldout_v2','--trial',str(trial)]
                proof=run_supervised(cmd,output/(clip+'.log'),output/(clip+'_resources.jsonl'),output/(clip+'_process_exit.json'),sample,300,15)
                if not proof['supervision_passed']:raise RuntimeError('media child failed')
                rp=output/clip/'receipt.json';item=json.loads(rp.read_text());video=output/clip/'video.mp4'
                if not (item['status']=='RECORDED_ISAACLAB_POSE_VIDEO_CREATED' and item['output_writers_closed']
                        and item['new_physics_steps']==item['new_optimizer_updates']==item['sdk_runs']==0
                        and item['nvidia_libraries_loaded']==[] and 'llvmpipe' in item['gl_renderer'].lower()
                        and item['frames_encoded']==item['frames_decoded']==200 and item['fps']==50
                        and sha(video)==item['video_sha256'] and video.stat().st_size==item['video_bytes']):
                    raise ValueError('video output not accepted')
                rec['completed_clips'].append(dict(clip=clip,trial_in_finish_order=trial,receipt_sha256=sha(rp),
                    process_exit_sha256=sha(output/(clip+'_process_exit.json')),resources_sha256=sha(output/(clip+'_resources.jsonl')),
                    video_path=str(video),video_sha256=sha(video),video_bytes=video.stat().st_size,
                    frames=item['frames_decoded'],provenance=item['provenance']))
                save()
            rec.update(status='MIXED1000_EXECUTION_POSE_VIDEOS_FINISHED',output_writers_closed=True);save()
        except BaseException as e:
            rec.update(status='MIXED1000_EXECUTION_POSE_VIDEOS_FAILED',error=repr(e),traceback=traceback.format_exc());save();raise

if __name__=='__main__':main()
