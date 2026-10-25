"""One private reference through the actual paper-configured Tracker MDP.

Eight environments, an untrained seeded actor,24 control calls, no optimizer.
This validates reference layout/observations/rewards, not a tracking result.
"""
import argparse,hashlib,json,pathlib,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--reference-witness',required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_updates':0,'physical_filter_executed':False,'training_started':False,
         'tracking_success_evaluated':False,'output_writers_closed':False};app=None;env=None

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2))

try:
    spec=json.loads(pathlib.Path(a.spec).read_text());assert a.headless and a.device=='cuda:1'
    app=AppLauncher(a).app
    import numpy as np,torch,gymnasium as gym
    from rsl_rl.modules import ActorCritic
    from paper_filter_config import make,paper_settings
    torch.set_num_threads(4);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
    torch.cuda.set_per_process_memory_fraction(.65,torch.device(a.device))
    witness=pathlib.Path(a.reference_witness);meta=json.loads((witness/'receipt.json').read_text())
    assert meta['status']=='TEXTOP_TRACKER_PHYSICS_KERNEL_PASSED'
    original=next(json.loads(x) for x in (pathlib.Path(spec['root'])/'data_preparation_20261025/available500_v1/prepared_manifest.private.jsonl').read_text().splitlines()
                  if json.loads(x)['output_sha256']==meta['private_reference_sha256'])
    with np.load(witness/'reference_fk.private.npz',allow_pickle=False) as f:
        q=f['qpos'].copy();body_pos=f['body_pos_w'].copy();body_quat=f['body_quat_w'].copy()
    order=[original['joint_names'].index(n) for n in meta['joint_names']];jp=q[:,7:][:,order]
    prev=body_quat[:-2];nxt=body_quat[2:]
    dw=np.sum(prev*nxt,axis=-1);dv=prev[...,:1]*nxt[...,1:]-nxt[...,:1]*prev[...,1:]+np.cross(prev[...,1:],nxt[...,1:])
    flip=np.where(dw<0,-1.,1.);dw*=flip;dv*=flip[...,None]
    norm=np.linalg.norm(dv,axis=-1);omega=dv*(2*np.arctan2(norm,dw)/np.maximum(norm,1e-12))[...,None]/.04
    omega=np.concatenate([omega[:1],omega,omega[-1:]],axis=0)
    motion=out/'motion.private.npz'
    np.savez_compressed(motion,fps=np.array([50]),joint_pos=jp,joint_vel=np.gradient(jp,.02,axis=0),
        body_pos_w=body_pos,body_quat_w=body_quat,body_lin_vel_w=np.gradient(body_pos,.02,axis=0),body_ang_vel_w=omega)
    cfg,agent=make(spec,[motion],out/'usd',phase='relaxed',num_envs=8)
    strict,_=make(spec,[motion],out/'strict_usd',phase='strict',num_envs=8)
    assert cfg.terminations.anchor_pos is None and cfg.terminations.ee_body_pos is None
    assert strict.terminations.anchor_pos.params['threshold']==.25 and strict.terminations.ee_body_pos.params['threshold']==.5
    assert all(getattr(cfg.events,n) is None for n in ['physics_material','add_joint_default_pos','base_com','push_robot'])
    env=gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg);obs,_=env.reset()
    assert obs['policy'].shape==(8,557) and obs['critic'].shape==(8,557)
    actor=ActorCritic(557,557,29,actor_hidden_dims=[2048,1024,512],critic_hidden_dims=[2048,1024,512],activation='elu').to(a.device)
    rewards=[];terminations=0
    with torch.no_grad():
        for _ in range(24):
            action=actor.act_inference(obs['policy']);obs,reward,terminated,truncated,info=env.step(action)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(obs['critic']).all() and torch.isfinite(reward).all()
            rewards.append(float(reward.mean()));terminations+=int(terminated.sum())
    receipt.update(status='TEXTOP_TRACKER_ENVIRONMENT_PROBE_PASSED',num_envs=8,control_steps=24,physics_steps=96,
        policy_observation_dim=557,critic_observation_dim=557,reference_frames=len(q),reference_joint_dim=jp.shape[1],
        reference_body_count=body_pos.shape[1],reference_sha256=sha(motion),paper_settings=paper_settings(),
        reward_means=rewards,terminated_environment_samples=terminations,source_sha256=sha(__file__),
        config_source_sha256=sha(pathlib.Path(__file__).with_name('paper_filter_config.py')),
        output_writers_closed=True,full_filter_training_ready=False)
    save();print('TEXTOP_TRACKER_ENVIRONMENT_PROBE_PASSED',flush=True)
except BaseException as error:
    receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
    if app is not None:
        from bounded_headless_close import close
        close(app,receipt,save,out,simulation=env.unwrapped.sim if env is not None else None,
              successful_statuses=('TEXTOP_TRACKER_ENVIRONMENT_PROBE_PASSED',))
