"""Read-only first-attempt CLoSD diagnostic. No model/reward/reset changes.

Record the generated reference supplied to the policy observation BEFORE its
action. Compare with the refreshed physical state after that same step.
Keep failed prefixes, unlike the official successful-episode-only recorder.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--envs', type=int, default=64)
p.add_argument('--seed', type=int, default=11)
args = p.parse_args()
out = Path(args.output).resolve()
assert not out.exists(), 'Choose a fresh output; retain prior attempts.'
out.mkdir(parents=True)
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gap_closd_entry', root/'scripts/r3_closd_isolated_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

class CollectionComplete(Exception):
    pass

started = time.time()
store = {}
original_create = entry.official.create_rlgpu_env

def create(*a, **kw):
    env = original_create(*a, **kw)
    task = env.task
    n, b, max_steps = task.num_envs, task.num_bodies, 315
    assert abs(float(task.dt)-1/30) < 1e-6, 'Unexpected control dt'
    store.update(
        actual=np.zeros((max_steps,n,b,3),np.float32),
        reference=np.zeros((max_steps,n,b,3),np.float32),
        contact_force=np.zeros((max_steps,n,b,3),np.float32),
        valid=np.zeros((max_steps,n),bool),
        progress=np.zeros((max_steps,n),np.int32),
        boundary=np.zeros(max_steps,bool),
        finished=np.zeros(n,bool), terminated=np.zeros(n,bool),
        lengths=np.zeros(n,np.int32), prompts=[None]*n, db_keys=[None]*n,
        names=list(task._body_names), dt=float(task.dt), horizon=int(task.planning_horizon_30fps),
        steps=0, pending=None)
    # The inherited reward reads a SEPARATE dataset cache, not the generated
    # reference in the policy observation. Never use that cache for this metric.
    observe = task._compute_task_obs_demo
    store['observation_ref_frame'] = int(task.frame_idx)
    def capture_observation(*a, **kw):
        result = observe(*a, **kw)
        env_ids = kw.get('env_ids', a[0] if a else None)
        if env_ids is None or len(env_ids)==task.num_envs:
            store['observation_ref_frame'] = int(task.frame_idx)
        return result
    task._compute_task_obs_demo = capture_observation
    pre = task.pre_physics_step
    store['applied_reference'] = None
    def capture_pre(actions):
        store['applied_reference'] = task.ref_body_pos.detach().cpu().numpy().copy()
        store['applied_ref_frame'] = store['observation_ref_frame']
        return pre(actions)
    task.pre_physics_step = capture_pre
    reward = task._compute_reward
    def capture_reward(actions):
        result = reward(actions)
        assert store['applied_reference'] is not None
        store['pending'] = (
            task._rigid_body_pos.detach().cpu().numpy().copy(),
            store['applied_reference'],
            task._contact_forces.detach().cpu().numpy().copy(),
            task.progress_buf.detach().cpu().numpy().copy(),
            store['applied_ref_frame'])
        return result
    task._compute_reward = capture_reward
    post = task.post_physics_step
    def capture_post():
        result = post()
        k = store['steps']
        assert store['pending'] is not None
        actual, ref, contact, progress, frame = store['pending']
        active = ~store['finished']
        assert np.isfinite(actual[active]).all() and np.isfinite(ref[active]).all()
        store['actual'][k] = actual
        store['reference'][k] = ref
        store['contact_force'][k] = contact
        store['progress'][k] = progress
        store['valid'][k] = active
        store['boundary'][k] = frame % store['horizon'] == 0
        store['lengths'][active] += 1
        for i in np.flatnonzero(active):
            if store['prompts'][i] is None:
                store['prompts'][i] = str(task.hml_prompts[i])
                store['db_keys'][i] = str(task.db_keys[i])
        done = task.reset_buf.detach().cpu().numpy().astype(bool) & active
        term = task._terminate_buf.detach().cpu().numpy().astype(bool) & done
        store['terminated'] |= term
        store['finished'] |= done
        store['steps'] += 1
        store['pending'] = None
        if store['steps'] % 60 == 0:
            print('GAP_DIAGNOSTIC_PROGRESS', json.dumps(dict(seed=args.seed,
                steps=store['steps'], finished=int(store['finished'].sum()),
                terminated=int(store['terminated'].sum()), elapsed=time.time()-started)), flush=True)
        if store['finished'].all():
            raise CollectionComplete()
        if store['steps'] >= max_steps:
            raise RuntimeError('First episode did not finish within bounded 315 steps')
        return result
    task.post_physics_step = capture_post
    return env
entry.official.create_rlgpu_env = create

overrides = ['learning=im_big','robot=smpl_humanoid','env=closd_t2m',
    'exp_name=CLoSD_t2m_finetune','epoch=-1','test=True',
    'no_virtual_display=True','headless=True','no_log=True','debug=False',
    'seed='+str(args.seed),'env.num_envs='+str(args.envs),'env.episode_length=300',
    'env.save_motion.save_hml_episodes=False','learning.params.config.player.games_num=10000000']
with initialize_config_dir(version_base=None, config_dir=str(entry.SOURCE/'closd/data/cfg')):
    cfg = compose(config_name='config', overrides=overrides)
(out/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg),encoding='utf-8')
complete = False
try:
    entry.official.main.__wrapped__(cfg)
    raise RuntimeError('Runner returned before first-attempt collection completed')
except CollectionComplete:
    complete = True
finally:
    if torch.cuda.is_available(): torch.cuda.synchronize()
    for task in entry.captured:
        if getattr(task,'viewer',None) is not None:
            task.gym.destroy_viewer(task.viewer); task.viewer=None
        if getattr(task,'sim',None) is not None:
            task.gym.destroy_sim(task.sim); task.sim=None
    print('GAP_NATIVE_CLEANUP_COMPLETE',flush=True)
if complete:
    steps=store['steps']
    fields=['actual','reference','contact_force','valid','progress','boundary']
    arrays={f:store[f][:steps] for f in fields}
    arrays.update(terminated=store['terminated'],lengths=store['lengths'])
    np.savez_compressed(out/'first_attempts.npz',**arrays)
    digest=hashlib.sha256((out/'first_attempts.npz').read_bytes()).hexdigest()
    receipt=dict(status='complete',seed=args.seed,environments=args.envs,
        steps=steps,dt=store['dt'],horizon=store['horizon'],body_names=store['names'],
        prompts=store['prompts'],db_keys=store['db_keys'],
        terminated=int(store['terminated'].sum()),
        completed_without_termination=int((~store['terminated']).sum()),
        seconds=time.time()-started,bytes=(out/'first_attempts.npz').stat().st_size,
        sha256=digest,source_commit='d8950bdb78312fcd2844f336b3fc079e7368c87f',
        checkpoint='official CLoSD_t2m_finetune; released DiP_no-target',
        model_updated=False,policy_updated=False,all_first_attempts_saved=True,
        alignment='v3: generated observation reference captured BEFORE action; actual AFTER same physics step',
        limitations=['SMPL simulated character, not a hardware-constrained humanoid robot',
            'Text-to-motion seeded by official HumanML prefix; no spatial target inputs',
            'Net foot-body contact forces do not give exact surface contact points'])
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('GAP_DIAGNOSTIC_PASS',json.dumps({k:receipt[k] for k in
        ['seed','environments','steps','terminated','seconds','sha256']}),flush=True)
