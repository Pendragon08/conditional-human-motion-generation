"""Bounded Isaac Gym camera recording of released CLoSD policies.

Inference only. Uses the pinned official task, DiP and player. Instrumentation
enables the graphics device, camera sensors, read-only traces and a bounded
recording stop. It does not estimate the published 4096-environment scores.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--frames', type=int, default=900)
opts, overrides = parser.parse_known_args()
assert opts.frames >= 2
out = Path(opts.output).resolve()
out.mkdir(parents=True, exist_ok=True)
assert not (out/'manifest.json').exists(), 'Do not overwrite a completed recording'
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r3_closd_entry', root/'scripts/r3_closd_isolated_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)  # Isaac Gym is imported before Torch here.
from isaacgym import gymapi
import cv2
import numpy as np
from closd.env.tasks.base_task import BaseTask
from closd.utils.closd_util import STATES
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

old_init = BaseTask.__init__
def camera_init(self, cfg, enable_camera_sensors=False):
    return old_init(self, cfg, enable_camera_sensors=True)
BaseTask.__init__ = camera_init
def camera_only_viewer(self):
    # Use one sensor per recorded environment, with no unused default sensors.
    assert self.headless
    self.viewer = None
    self.recorder_camera_handles = []
    self.recorder_camera_handle = -1
    self.recording = self.recording_state_change = False
BaseTask.create_viewer = camera_only_viewer

class RecordingFinished(Exception):
    pass

writers = {}
trace = None
videos = []
created_cameras = []
count = 0
started = time.time()
old_create = entry.official.create_rlgpu_env
def create(*args, **kwargs):
    global trace
    env = old_create(*args, **kwargs)
    task = env.task
    assert task.graphics_device_id == 0
    ids = {}
    for state in (STATES.REACH, STATES.STRIKE_KICK, STATES.STRIKE_PUNCH, STATES.SIT):
        matches = (task.init_state_per_env == state).nonzero().flatten().tolist()
        assert matches, 'Fixed seed/batch has no {}'.format(state.name)
        ids[state.name] = matches[0]  # fixed first environment; no success selection
    fps = 1.0 / (task.sim_params.dt * task.control_freq_inv)
    cams = {}
    width, height = 640, 480
    for name, idx in ids.items():
        props = gymapi.CameraProperties()
        props.width, props.height = width, height
        props.horizontal_fov = 65.0
        cam = task.gym.create_camera_sensor(task.envs[idx], props)
        assert cam >= 0
        cams[name] = cam
        created_cameras.append((task, idx, cam))
        video = out/(name.lower()+'.mp4')
        assert not video.exists(), 'Preserve previous partial videos in a different output'
        err = open(out/(name.lower()+'.ffmpeg.log'), 'w')
        proc = subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','warning',
            '-f','rawvideo','-pix_fmt','rgb24','-s','{}x{}'.format(width,height),
            '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
            '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
            stdin=subprocess.PIPE, stderr=err)
        writers[name] = (proc, err)
        videos.append({'task':name,'environment_id':idx,'file':video.name,
            'fps':fps,'width':width,'height':height,
            'includes_get_up_transition':name=='SIT'})
    trace = open(out/'actual_state_trace.jsonl', 'w')
    original_post = task.post_physics_step
    def post():
        global count
        result = original_post()
        count += 1
        task.gym.fetch_results(task.sim, True)
        task.gym.step_graphics(task.sim)
        state_rows = []
        for name, idx in ids.items():
            human = task._humanoid_root_states[idx,:3].detach().cpu().numpy()
            target = task._target_states[idx,:3].detach().cpu().numpy()
            # Same env-local coordinates used by the official follower camera.
            center = (human+target)/2
            task.gym.set_camera_location(cams[name],task.envs[idx],
                gymapi.Vec3(float(center[0]+4),float(center[1]-5),3.0),
                gymapi.Vec3(float(center[0]),float(center[1]),0.9))
            state_rows.append({'task':name,'env':idx,
                'state':STATES(int(task.cur_state[idx].item())).name,
                'done':bool(task.is_done[idx].item()),
                'reset':bool(task.reset_buf[idx].item()),
                'prompt':str(task.hml_prompts[idx]),'root_xyz':human.tolist()})
        task.gym.render_all_camera_sensors(task.sim)
        for name, idx in ids.items():
            rgba = task.gym.get_camera_image(task.sim,task.envs[idx],cams[name],gymapi.IMAGE_COLOR)
            rgb = np.asarray(rgba).reshape(height,width,4)[:,:,:3].copy()
            assert rgb.std() > 1, 'Camera image is blank'
            writers[name][0].stdin.write(rgb.tobytes())
            if count in (1,min(30,opts.frames),opts.frames):
                cv2.imwrite(str(out/('{}_{:04d}.jpg'.format(name.lower(),count))),
                    cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        trace.write(json.dumps({'step':count,'simulation_seconds':count/fps,
            'wall_time':time.time(),'source':'actual_isaacgym_camera','states':state_rows})+'\n')
        if count % 30 == 0:
            trace.flush()
            print('CLOSD_RECORDING_PROGRESS',count,flush=True)
        if count >= opts.frames:
            raise RecordingFinished()
        return result
    task.post_physics_step = post
    return env
entry.official.create_rlgpu_env = create

with initialize_config_dir(version_base=None,config_dir=str(entry.SOURCE/'closd/data/cfg')):
    cfg = compose(config_name='config',overrides=overrides)
assert cfg.test and not cfg.closd_eval and cfg.headless
assert cfg.env.num_envs == 12 and cfg.seed == 42
assert cfg.env.dip.cfg_param == 7.5
model = entry.SOURCE/'output/CLoSD'/cfg.exp_name/'Humanoid.pth'
assert model.is_file()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checkpoint_sha = sha(model)
OmegaConf.save(cfg, out/'resolved_config.yaml')
complete = False
try:
    entry.official.main.__wrapped__(cfg)
    raise RuntimeError('Official player returned before requested recording length')
except RecordingFinished:
    complete = True
    print('CLOSD_RECORDING_BOUND_REACHED',count,flush=True)
finally:
    for proc, err in writers.values():
        proc.stdin.close()
        rc = proc.wait(timeout=60)
        err.close()
        if rc != 0:
            complete = False
            print('CLOSD_ENCODER_FAILED',rc,flush=True)
    if trace:
        trace.close()
    entry.torch.cuda.synchronize()
    print('CLOSD_NATIVE_CLEANUP_BEGIN',flush=True)
    for task in entry.captured:
        # Release all camera resources before the graphics-enabled simulator.
        for owned_task, idx, cam in created_cameras:
            if owned_task is task:
                task.gym.destroy_camera_sensor(task.sim, task.envs[idx], cam)
        for idx, cam in enumerate(getattr(task,'recorder_camera_handles',[])):
            if cam >= 0:
                task.gym.destroy_camera_sensor(task.sim, task.envs[idx], cam)
        task.recorder_camera_handles = []
        if getattr(task,'viewer',None) is not None:
            task.gym.destroy_viewer(task.viewer); task.viewer=None
        if getattr(task,'sim',None) is not None:
            task.gym.destroy_sim(task.sim); task.sim=None
    print('CLOSD_NATIVE_CLEANUP_COMPLETE',flush=True)
assert complete and count == opts.frames
for item in videos:
    p = out/item['file']
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,width,height,duration,r_frame_rate',
        '-of','json',str(p)]))['streams'][0]
    assert int(probe['nb_frames']) == opts.frames
    assert float(probe['duration']) > 0
    item.update({'sha256':sha(p),'ffprobe':probe,'bytes':p.stat().st_size})
assert sha(model) == checkpoint_sha
manifest = {'status':'COMPLETE','kind':'frozen_policy_demonstration',
    'source':'actual_isaacgym_camera','checkpoint':str(model),'checkpoint_sha256':checkpoint_sha,
    'experiment':cfg.exp_name,'seed':42,'num_envs':12,'frames':count,
    'cfg':7.5,'episode_length':int(cfg.env.episode_length),
    'official_source_commit':'d8950bdb78312fcd2844f336b3fc079e7368c87f',
    'elapsed_seconds':time.time()-started,'normal_exit':None,
    'evaluation_video_verified':False,
    'note':'Post-hoc demonstration; not a recording of the completed 4096-env evaluation. SIT includes FSM GET_UP when reached; inspect state trace.',
    'videos':videos,'trace_sha256':sha(out/'actual_state_trace.jsonl')}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('CLOSD_RECORDING_COMPLETE',json.dumps(manifest),flush=True)
