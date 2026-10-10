"""Unmodified CLoSD T2M policy and generator; bounded recording and provenance.

The official recorder retains full non-terminated episodes only. We separately
count termination/reset events, so its saved motion quality is not mistaken for
an unconditional task success rate. No training, reward or model changes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--envs', type=int, required=True)
p.add_argument('--episodes', type=int, required=True)
p.add_argument('--seed', type=int, default=10)
args = p.parse_args()
out = Path(args.output).resolve()
assert not out.exists(), 'Preserve all prior attempts; choose a fresh output path.'
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline_closd_entry', root/'scripts/r3_closd_isolated_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

class RecordingComplete(Exception):
    pass

started = time.time()
counts = {'steps': 0, 'reset_events': 0, 'termination_events': 0, 'full_valid_events': 0}
original_create = entry.official.create_rlgpu_env
def create(*a, **kw):
    env = original_create(*a, **kw)
    task = env.task
    original_post = task.post_physics_step
    def post():
        result = original_post()
        counts['steps'] += 1
        counts['reset_events'] += int(task.reset_buf.sum().item())
        counts['termination_events'] += int(task._terminate_buf.sum().item())
        counts['full_valid_events'] += len(task.get_episode_ids())
        if counts['steps'] % 100 == 0:
            print('CLOSD_FRESH_PROGRESS', json.dumps(dict(counts, saved=task.n_all_episodes,
                  elapsed_seconds=time.time()-started)), flush=True)
        if task.done_saving_episodes:
            assert task.n_all_episodes == args.episodes
            counts['saved_episodes'] = task.n_all_episodes
            raise RecordingComplete()
        return result
    task.post_physics_step = post
    return env
entry.official.create_rlgpu_env = create
overrides = [
    'learning=im_big', 'robot=smpl_humanoid', 'env=closd_t2m',
    'exp_name=CLoSD_t2m_finetune', 'epoch=-1', 'test=True',
    'no_virtual_display=True', 'headless=True', 'no_log=True',
    'debug=False', 'seed='+str(args.seed), 'env.num_envs='+str(args.envs),
    'env.episode_length=300', 'env.save_motion.save_hml_episodes=True',
    'env.save_motion.save_hml_episodes_dir='+str(out),
    'env.save_motion.max_saved_hml_episodes='+str(args.episodes),
    'learning.params.config.player.games_num=10000000'
]
with initialize_config_dir(version_base=None, config_dir=str(entry.SOURCE/'closd/data/cfg')):
    cfg = compose(config_name='config', overrides=overrides)
completed = False
try:
    entry.official.main.__wrapped__(cfg)
    raise RuntimeError('Official runner returned before the requested motion collection completed.')
except RecordingComplete:
    completed = True
finally:
    torch.cuda.synchronize()
    for task in entry.captured:
        if getattr(task, 'viewer', None) is not None:
            task.gym.destroy_viewer(task.viewer); task.viewer = None
        if getattr(task, 'sim', None) is not None:
            task.gym.destroy_sim(task.sim); task.sim = None
    print('CLOSD_NATIVE_CLEANUP_COMPLETE', flush=True)
if completed:
    motion = out/'CLoSD.pkl'
    digest = hashlib.sha256()
    with motion.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(chunk)
    receipt = dict(counts, status='complete', elapsed_seconds=time.time()-started,
        output_bytes=motion.stat().st_size, sha256=digest.hexdigest(), seed=args.seed,
        environments=args.envs, source_commit='d8950bdb78312fcd2844f336b3fc079e7368c87f',
        checkpoint='official released CLoSD_t2m_finetune', training_performed=False,
        selection='Official full non-terminated episode filter; not an unconditional success rate',
        adaptations=['serial reference loading', 'bounded stop after official save',
                     'read-only event counters', 'explicit native resource cleanup'])
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('CLOSD_FRESH_RECORDING_PASS', json.dumps(receipt), flush=True)
