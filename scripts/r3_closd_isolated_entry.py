"""Official CLoSD entry with lifecycle diagnostics and one-way scene telemetry.

No optimizer/model/reward/sampling overrides. Renderer imports never enter PPO.
"""
import faulthandler
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

faulthandler.enable(all_threads=True)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'third_party/CLoSD'
sys.path.insert(0, str(SOURCE/'closd'))
sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)
spec = importlib.util.spec_from_file_location('r3_official_closd_run', SOURCE/'closd/run.py')
official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
import torch
from closd.env.tasks.humanoid import Humanoid
from closd.utils.flags import flags
from closd.utils.motion_lib_base import MotionLibBase

captured = []
asset_paths = {}
original_xml = Humanoid._create_smpl_humanoid_xml
def capture_xml(self, *args, **kwargs):
    result = original_xml(self, *args, **kwargs)
    if isinstance(result, dict) and 0 in result: asset_paths[id(self)] = str(Path(result[0][1]).resolve())
    return result
Humanoid._create_smpl_humanoid_xml = capture_xml

# Same documented serial-loading adaptation as the previously verified runtime.
# Restore the flag immediately; PPO and environment debug remain False.
original_load = MotionLibBase.load_motions
def serial_load(self, *args, **kwargs):
    previous = flags.debug
    try:
        flags.debug = True
        return original_load(self, *args, **kwargs)
    finally: flags.debug = previous
MotionLibBase.load_motions = serial_load

original_create = official.create_rlgpu_env
def create(*args, **kwargs):
    env = original_create(*args, **kwargs)
    task = env.task
    captured.append(task)
    endpoint = os.environ.get('R3_CLOSD_SCENE_UDP')
    if endpoint:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.setblocking(False)
        host, port = endpoint.rsplit(':', 1)
        assert host == '127.0.0.1'
        names=[]; assets=[]; origin=task.gym.get_env_origin(task.envs[0])
        for actor in range(task.gym.get_actor_count(task.envs[0])):
            names.append(task.gym.get_actor_rigid_body_names(task.envs[0], actor))
            if actor == 0: assets.append(asset_paths.get(id(task)))
            else:
                state=str(task.init_state_per_env[0].item()) if hasattr(task,'init_state_per_env') else ''
                # State enum is recorded; actual first-environment target asset
                # is selected from its initialized state below.
                from closd.utils.closd_util import STATES
                state_value=task.init_state_per_env[0].item()
                filename='sofa.urdf' if state_value==STATES.SIT else (
                    'location_marker.urdf' if state_value==STATES.REACH else 'strike_target.urdf')
                assets.append(str(SOURCE/'closd/data/assets/urdf'/filename))
        meta={'source':'actual_isaacgym_simulation','assets':assets,'body_names':names,
              'origin':[origin.x,origin.y,origin.z],'experiment':official.cfg.exp_name}
        meta_path=ROOT/'artifacts/monitoring/r3_closd_actual_scene_meta.json'
        tmp=meta_path.with_suffix('.tmp'); tmp.write_text(json.dumps(meta)); tmp.replace(meta_path)
        original_post=task.post_physics_step
        counter=[0]; warned=[False]
        def post():
            result=original_post(); counter[0]+=1
            if counter[0]%10==0:
                try:
                    poses=task._rigid_body_state.view(task.num_envs,-1,13)[0,:,:7].detach().cpu().tolist()
                    packet={'source':'actual_isaacgym_simulation','ts':time.time(),'step':counter[0],
                            'experiment':official.cfg.exp_name,'poses_xyzw':poses,
                            'prompt':str(task.hml_prompts[0])}
                    sock.sendto(json.dumps(packet).encode(),(host,int(port)))
                except Exception as error:
                    if not warned[0]: print('CLOSD_SCENE_SEND_WARNING',repr(error),flush=True); warned[0]=True
            return result
        task.post_physics_step=post
    return env
official.create_rlgpu_env=create

if __name__ == '__main__':
    from hydra import compose, initialize_config_dir
    config_only = '--resolved-config-only' in sys.argv
    overrides = [v for v in sys.argv[1:] if v != '--resolved-config-only']
    with initialize_config_dir(version_base=None, config_dir=str(SOURCE/'closd/data/cfg')):
        configuration = compose(config_name='config', overrides=overrides)
    if config_only:
        print('CLOSD_CONFIG_PROBE_PASS',configuration.env.num_envs,configuration.headless,
              configuration.env.dip.cfg_param,flush=True)
        raise SystemExit(0)
    completed=False
    try:
        official.main.__wrapped__(configuration)
        completed=True
        print('CLOSD_OFFICIAL_RUNNER_RETURNED',flush=True)
    except SystemExit as error:
        # The unmodified official evaluator calls exit() after its final table.
        # Preserve nonzero exits and training exits as failures.
        if configuration.closd_eval and error.code in (None, 0):
            completed=True
            print('CLOSD_OFFICIAL_EVALUATOR_EXIT_ZERO',flush=True)
        else:
            raise
    finally:
        if completed:
            torch.cuda.synchronize()
            print('CLOSD_NATIVE_CLEANUP_BEGIN',flush=True)
            for task in captured:
                if getattr(task,'viewer',None) is not None:
                    task.gym.destroy_viewer(task.viewer); task.viewer=None
                if getattr(task,'sim',None) is not None:
                    task.gym.destroy_sim(task.sim); task.sim=None
            print('CLOSD_NATIVE_CLEANUP_COMPLETE',flush=True)
