"""Verify supplementary CLoSD videos and build shareable, pose-free provenance."""
import collections, gzip, hashlib, itertools, json, math, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO=Path(r'D:\具身智能\conditional-human-motion-generation')
ROOT=Path(r'D:\具身智能\通用条件人体运动生成_已完成实验结果_20260929')
SOURCE=REPO/'artifacts/closd_recordings_20261006/released_pose_pair_v1'
MEDIA=REPO/'media/closd_released_20261006'
RESULT=REPO/'results/closd_supplementary_20261006'
LOCAL=ROOT/'CLoSD补充结果_20261006'
FFMPEG=REPO/'artifacts/report_dependencies/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
RESULT.mkdir(exist_ok=True);LOCAL.mkdir(exist_ok=True)
for sub in ['视频','证据','训练曲线']:(LOCAL/sub).mkdir(exist_ok=True)

def verify_video(p,frames):
    probe=subprocess.run([str(FFMPEG),'-hide_banner','-i',str(p)],capture_output=True,text=True,encoding='utf-8',errors='replace')
    info=probe.stderr
    assert 'h264' in info and '640x480' in info and '30 fps' in info,info
    decoded=subprocess.run([str(FFMPEG),'-v','error','-i',str(p),'-progress','pipe:1','-f','null','-'],capture_output=True,text=True,timeout=120)
    assert decoded.returncode==0 and not decoded.stderr,decoded.stderr
    count=int(re.findall(r'^frame=(\d+)$',decoded.stdout,re.M)[-1]);assert count==frames,(p,count,frames)
    return {'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'frames':count,
            'fps':30,'duration_seconds':count/30,'width':640,'height':480,'codec':'h264',
            'full_decode_exit_code':0,'video_verified':True}

summary={'created_at':datetime.now(timezone.utc).isoformat(),'kind':'supplementary_frozen_policy_demonstrations',
 'official_source_commit':'d8950bdb78312fcd2844f336b3fc079e7368c87f',
 'numeric_official_eval_results_changed':False,'original4096_evaluation_video_verified':False,
 'supplementary_source':'Actual Isaac Gym rigid-body/object states rendered independently with MeshCat',
 'physics_frames_interpolated':False,'simulation_engine_changed':False,'ppo_updates':0,
 'render_coordinate_convention':'Gym tensors are environment-local; do not subtract environment origin again. MeshCat Z-up scene/camera conversion is rendering only.',
 'selection':'First fixed environment of each initialized task, no success filtering; GET_UP excerpt is explicitly selected by longest contiguous recorded state interval.',
 'formal3072_training_started':False,'local_scene_gate_passed':False,
 'latest_user_local_scene_response':'画面不动或打不开','models':[]}
for model in ['CLoSD_multitask_finetune','CLoSD_no_finetune']:
    directory=SOURCE/model;manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['normal_exit'] is True and manifest['exit_code']==0
    assert manifest['pose_sha256']==sha(directory/'actual_poses.json.gz')
    data=json.load(gzip.open(directory/'actual_poses.json.gz','rt'));assert len(data['frames'])==900
    assert [f['step'] for f in data['frames']]==list(range(1,901))
    for frame in data['frames']:
        assert abs(frame['simulation_seconds']-frame['step']/30)<1e-4
        for environment in frame['environments']:
            assert len(environment['poses_xyzw'])==25
            for pose in environment['poses_xyzw']:
                assert len(pose)==7 and all(math.isfinite(x) for x in pose)
                assert abs(sum(x*x for x in pose[3:])-1)<1e-3
    entry={'model':model,'run':manifest,'clips':[],'state_frame_counts':{},'task_environment_ids':{}}
    trace=[]
    for i,meta in enumerate(data['metadata']):
        states=[f['environments'][i]['state'] for f in data['frames']]
        entry['state_frame_counts'][meta['task']]=dict(collections.Counter(states))
        entry['task_environment_ids'][meta['task']]=meta['environment_id']
        p=MEDIA/(model+'_'+meta['task']+'.mp4');v=verify_video(p,900)
        v.update(task=meta['task'],clip_kind='complete_fixed_environment_recording',first_source_step=1,last_source_step=900)
        entry['clips'].append(v)
    for f in data['frames']:
        trace.append({'step':f['step'],'simulation_seconds':f['simulation_seconds'],
          'environments':[{k:e[k] for k in ['task','state','done','reset','prompt']} for e in f['environments']]})
    # No rigid-body transforms, XML/URDF scene geometry or checkpoint data in shareable evidence.
    (RESULT/(model+'_state_trace.json')).write_text(json.dumps(trace,ensure_ascii=False),encoding='utf-8')
    if model=='CLoSD_multitask_finetune':
        spans=[];start=0
        for state,group in itertools.groupby(f['environments'][3]['state'] for f in data['frames']):
            length=len(list(group))
            if state=='GET_UP':spans.append((start,length))
            start+=length
        start,length=max(spans,key=lambda p:p[1]);p=MEDIA/(model+'_GET_UP_excerpt.mp4')
        if not p.exists():
            cmd=[str(FFMPEG),'-hide_banner','-loglevel','error','-i',str(MEDIA/(model+'_SIT.mp4')),
                 '-vf',f'trim=start_frame={start}:end_frame={start+length},setpts=N/(30*TB)',
                 '-an','-r','30','-fps_mode','cfr','-enc_time_base','1:30','-c:v','libx264',
                 '-pix_fmt','yuv420p','-crf','21','-movflags','+faststart',str(p)]
            subprocess.run(cmd,check=True,timeout=60)
        v=verify_video(p,length);v.update(task='GET_UP',clip_kind='longest_contiguous_state_excerpt',
          first_source_step=start+1,last_source_step=start+length,parent_video=model+'_SIT.mp4')
        entry['clips'].append(v)
    else:entry['get_up_demonstration']='Not triggered in the selected fixed SIT environment during900 frames; do not invent a successful GET_UP clip.'
    shutil.copy2(directory/'resolved_config.yaml',RESULT/(model+'_resolved_config.yaml'))
    summary['models'].append(entry)
    for clip in entry['clips']:
        shutil.copy2(MEDIA/clip['file'],LOCAL/'视频'/clip['file'])
    for p in MEDIA.glob(model+'*_300.jpg'):shutil.copy2(p,LOCAL/'视频'/p.name)

summary['complete_videos']=8;summary['derived_excerpt_videos']=1;summary['total_videos']=9
summary['native_camera_attempts']={'attempts':3,'actual_exit_codes':[139,139,139],
 'full_native_camera_job_started':False,'old512_exit139_root_cause_proven':False}
(RESULT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
logs=REPO/'artifacts/closd_recordings_20261006/log_source/artifacts/monitoring'
for p in logs.glob('*'):
    if p.suffix in {'.log','.state'}:shutil.copy2(p,RESULT/p.name)
for p in RESULT.iterdir():
    if p.is_file():shutil.copy2(p,LOCAL/'证据'/p.name)
for p in (REPO/'results/closd_old512_training_curves_20261006').iterdir():
    if p.is_file():shutil.copy2(p,LOCAL/'训练曲线'/p.name)
print(json.dumps({'videos':9,'models':2,'original_eval_video_verified':False,'destination':str(LOCAL)},ensure_ascii=False))
