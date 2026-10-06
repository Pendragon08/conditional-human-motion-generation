"""Export real frozen-policy Isaac Gym states without a graphics context.

Render these separately; no second simulator is used to calculate the states.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--frames',type=int,default=900)
args,overrides=p.parse_known_args()
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/'manifest.json').exists()
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r3_closd_entry',root/'scripts/r3_closd_isolated_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
from closd.utils.closd_util import STATES
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

class Finished(Exception): pass
all_xml={};old_xml=entry.Humanoid._create_smpl_humanoid_xml
def xml(self,*a,**k):
    result=old_xml(self,*a,**k)
    all_xml[id(self)]={int(i):str(Path(v[1]).resolve()) for i,v in result.items()}
    return result
entry.Humanoid._create_smpl_humanoid_xml=xml
old_create=entry.official.create_rlgpu_env
frames=[];meta=[]
def create(*a,**k):
    env=old_create(*a,**k);t=env.task
    assert t.graphics_device_id == -1, 'State export must not enable graphics'
    for state in (STATES.REACH,STATES.STRIKE_KICK,STATES.STRIKE_PUNCH,STATES.SIT):
        ids=(t.init_state_per_env==state).nonzero().flatten().tolist();assert ids
        idx=ids[0]
        filename='sofa.urdf' if state==STATES.SIT else (
            'location_marker.urdf' if state==STATES.REACH else 'strike_target.urdf')
        names=[t.gym.get_actor_rigid_body_names(t.envs[idx],i)
               for i in range(t.gym.get_actor_count(t.envs[idx]))]
        origin=t.gym.get_env_origin(t.envs[idx])
        meta.append({'task':state.name,'environment_id':idx,'body_names':names,
            'origin':[origin.x,origin.y,origin.z],
            'assets':[all_xml[id(t)].get(idx,all_xml[id(t)][0]),
                str(entry.SOURCE/'closd/data/assets/urdf'/filename)]})
    fps=1/(t.sim_params.dt*t.control_freq_inv)
    old_post=t.post_physics_step
    def post():
        result=old_post()
        poses=t._rigid_body_state.view(t.num_envs,-1,13)
        row={'step':len(frames)+1,'simulation_seconds':(len(frames)+1)/fps,
             'wall_time':time.time(),'environments':[]}
        for m in meta:
            idx=m['environment_id']
            row['environments'].append({'task':m['task'],
                'state':STATES(int(t.cur_state[idx].item())).name,
                'done':bool(t.is_done[idx].item()),'reset':bool(t.reset_buf[idx].item()),
                'prompt':str(t.hml_prompts[idx]),
                'poses_xyzw':poses[idx,:,:7].detach().cpu().tolist()})
        frames.append(row)
        if len(frames)%30==0:print('CLOSD_POSE_PROGRESS',len(frames),flush=True)
        if len(frames)>=args.frames:raise Finished()
        return result
    t.post_physics_step=post
    return env
entry.official.create_rlgpu_env=create
with initialize_config_dir(version_base=None,config_dir=str(entry.SOURCE/'closd/data/cfg')):
    cfg=compose(config_name='config',overrides=overrides)
assert cfg.test and not cfg.closd_eval and cfg.headless
assert cfg.env.num_envs==12 and cfg.seed==42 and cfg.env.dip.cfg_param==7.5
model=entry.SOURCE/'output/CLoSD'/cfg.exp_name/'Humanoid.pth';assert model.is_file()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
model_sha=sha(model);OmegaConf.save(cfg,out/'resolved_config.yaml')
ok=False;start=time.time()
try:
    entry.official.main.__wrapped__(cfg)
    raise RuntimeError('Unexpected early player return')
except Finished:
    ok=True;print('CLOSD_POSE_BOUND_REACHED',len(frames),flush=True)
finally:
    entry.torch.cuda.synchronize()
    print('CLOSD_NATIVE_CLEANUP_BEGIN',flush=True)
    for t in entry.captured:
        if t.viewer is not None:t.gym.destroy_viewer(t.viewer);t.viewer=None
        if t.sim is not None:t.gym.destroy_sim(t.sim);t.sim=None
    print('CLOSD_NATIVE_CLEANUP_COMPLETE',flush=True)
assert ok and len(frames)==args.frames and sha(model)==model_sha
with gzip.open(str(out/'actual_poses.json.gz'),'wt',encoding='utf-8') as f:
    json.dump({'source':'actual_isaacgym_simulation','metadata':meta,'frames':frames},f)
manifest={'status':'files_complete_process_exit_pending','kind':'frozen_policy_demonstration',
    'source':'actual_isaacgym_simulation_independent_renderer','experiment':cfg.exp_name,
    'checkpoint_sha256':model_sha,'seed':42,'num_envs':12,'frames':len(frames),
    'cfg':7.5,'episode_length':500,'normal_exit':None,
    'official_source_commit':'d8950bdb78312fcd2844f336b3fc079e7368c87f',
    'elapsed_seconds':time.time()-start,'pose_sha256':sha(out/'actual_poses.json.gz'),
    'evaluation_video_verified':False}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('CLOSD_RECORDING_COMPLETE',flush=True)
