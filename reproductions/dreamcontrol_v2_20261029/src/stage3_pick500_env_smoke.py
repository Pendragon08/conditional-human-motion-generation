"""Verify reference-bound randomized object reset with the real existing PhysX task."""
import argparse
import json
from pathlib import Path
import time
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.output_dir.mkdir(exist_ok=False)
app=AppLauncher(a).app
import torch
from isaaclab.envs import ManagerBasedRLEnv
import stage3_pick29_env as base
from stage3_pick29_core import sha
from stage3_pick500_env import Pick500EnvCfg,ReferenceBank500

env=None;began=time.monotonic()
try:
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.08,device=a.device)
    refused=False
    try:ReferenceBank500(a.manifest,sha(base.ASSET),a.device)
    except ValueError as e:
        assert 'not enough admitted' in str(e);refused=True
    assert refused,'four-candidate pilot cannot satisfy500-reference formal gate'
    cfg=Pick500EnvCfg();cfg.scene.num_envs=8;cfg.sim.device=a.device;cfg.seed=20261104
    cfg.reference_manifest=str(a.manifest);cfg.minimum_reference_count=2
    env=ManagerBasedRLEnv(cfg=cfg)
    maximum=0.;seen=set();leak=0.
    for reset in range(4):
        env.reset();env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
        ids=env._reference_ids.clone();seen.update(ids.tolist())
        desired=env._reference_bank.object_positions[ids]+env.scene.env_origins
        maximum=max(maximum,float((env.scene['object'].data.root_pos_w-desired).abs().max()))
        obs=base.live_observation(env).clone()
        env._reference_ids=(ids+1)%env._reference_bank.count
        leak=max(leak,float((base.live_observation(env)-obs).abs().max()))
        env._reference_ids=ids
        for step in range(8):
            actions=torch.zeros(8,31,device=a.device);actions[:,-2:]=1.
            observations,rewards,terminated,truncated,extras=env.step(actions)
            assert torch.isfinite(rewards).all() and torch.isfinite(base.live_observation(env)).all()
    assert maximum<1e-5 and leak==0. and len(seen)==env._reference_bank.count
    report={'status':'reference_object_reset_witness_passed','num_envs':8,'physics_control_steps':32,
        'available_pilot_references':env._reference_bank.count,'all_reference_ids_seen':sorted(seen),
        'maximum_object_reset_error_m':maximum,'actor_reference_change_error':leak,
        'body_dof':29,'policy_actions':31,'actor_observation':102,'formal500_count_rejected_pilot':refused,
        'formal_training_started':False,'task_success_claimed':False,'manifest_sha256':sha(a.manifest),
        'elapsed_seconds':time.monotonic()-began}
    with (a.output_dir/'report.json').open('x') as f:json.dump(report,f,indent=2)
    print('PICK500_OBJECT_RESET_WITNESS '+json.dumps(report),flush=True)
finally:
    if env is not None:env.close()
    app.close(wait_for_replicator=False)
