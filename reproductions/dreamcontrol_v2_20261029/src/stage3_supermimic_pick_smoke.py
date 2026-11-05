"""Real PhysX data -> SuperMimic-style model fitting/BPTT -> checkpoint witness.

This bounded run validates integration, not Pick task success or a full budget.
It uses the existing Isaac runtime and pinned TextOp SuperMimic primitives, so
the already calibrated PhysX robot, hands, object and reward remain identical.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
S=R/'stage3_pick29_20261103'
sys.path.insert(0,str(S/'src'))
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--num-envs',type=int,default=8)
parser.add_argument('--steps',type=int,default=128)
parser.add_argument('--model-updates',type=int,default=32)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
assert 2<=args.num_envs<=16 and 64<=args.steps<=160 and 8<=args.model_updates<=64
assert shutil.disk_usage('/data').free>8*1024**3+600*1024**2
args.output_dir.mkdir(exist_ok=False)
report={'status':'starting','pid':os.getpid(),'scope':'real bounded integration witness, not task or training-budget completion',
        'num_envs':args.num_envs,'real_vector_steps':0,'world_model_updates':0,'policy_updates':0}


def save():
    tmp=args.output_dir/'status.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False));tmp.replace(args.output_dir/'status.json')


save();app=None;env=None;began=time.monotonic()
try:
    app=AppLauncher(args).app
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.envs.mdp as mdp
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    import stage3_pick29_env as base
    from stage3_pick29_core import BODY_JOINTS,sha
    from stage3_resume_contract import equal_tree
    from stage3_supermimic_pick_core import (native_primitives,LivePolicy,LiveCritic,PickWorldModel,
        live_obs,hard_hands,private_reference,fit_world_model,improve_policy,delta_target,integrate)
    torch.set_num_threads(2);torch.manual_seed(20261104)
    torch.cuda.set_per_process_memory_fraction(.08,device=args.device)
    plan=json.loads((R/'stage3_supermimic29_20261103/handoff_plan.json').read_text())
    native=native_primitives(plan['existing_textop_environment']['source_root'])
    physical_ids=None

    def state_of(e):
        robot,obj=e.scene['robot'],e.scene['object']
        return torch.cat((robot.data.root_pos_w-e.scene.env_origins,robot.data.root_quat_w,
            robot.data.root_lin_vel_w,robot.data.root_ang_vel_w,robot.data.joint_pos[:,physical_ids],
            robot.data.joint_vel[:,physical_ids],obj.data.root_pos_w-e.scene.env_origins,
            obj.data.root_quat_w,obj.data.root_lin_vel_w,obj.data.root_ang_vel_w),-1)

    def capture(e):
        e._sm_next=state_of(e).clone()
        e._sm_reference=private_reference(base.reference(e)).clone()
        e._sm_terminated=base.fallen(e).clone()
        return mdp.is_alive(e)

    cfg=base.Pick29EnvCfg();cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=20261104
    cfg.reference_manifest=str(S/'reports/native_table_replay_final/reference_bank.json')
    cfg.rewards.alive.func=capture
    env=ManagerBasedRLEnv(cfg=cfg);wrapped=RslRlVecEnvWrapper(env)
    physical_names=BODY_JOINTS+sorted(n for n in env.scene['robot'].joint_names if n not in BODY_JOINTS)
    assert len(physical_names)==43 and len(set(physical_names))==43
    physical_ids=env.scene['robot'].find_joints(physical_names,preserve_order=True)[0]
    default_q=env.scene['robot'].data.default_joint_pos[0,physical_ids].clone()
    default_qd=env.scene['robot'].data.default_joint_vel[0,physical_ids].clone()
    obs,_=wrapped.reset()
    policy=LivePolicy(native).to(args.device);critic=LiveCritic(native).to(args.device)
    wm=PickWorldModel(native).to(args.device)
    policy_optimizer=torch.optim.Adam(policy.parameters(),lr=.001,betas=(.9,.95))
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=.001)
    model_optimizer=torch.optim.Adam(wm.parameters(),lr=.0002)
    data={k:[] for k in ['s','s1','a','a_prev','r','term','done','ref','step','id']}
    observation_error=0.
    report.update(status='collecting_real_transitions',state_dimension=112,physical_joint_count=43,
                  body_dof=29,actor_obs=102,policy_actions=31,actor_reference_input=False)
    save()
    for step in range(args.steps):
        with torch.no_grad():
            s=state_of(env);prev=env.action_manager.action.clone()
            derived=live_obs(s,prev,default_q,default_qd)
            observation_error=max(observation_error,float((derived-obs).abs().max()))
            assert observation_error<1e-4,('live observation does not match calibrated task',observation_error)
            policy.norm.update(obs)
            action=hard_hands(policy(obs)+.05*torch.randn_like(prev))
            steps=env.episode_length_buf.clone();ids=env._reference_ids.clone()
            obs,reward,done,extras=wrapped.step(action)
            values={'s':s,'s1':env._sm_next,'a':action,'a_prev':prev,'r':reward,
                    'term':env._sm_terminated,'done':done.bool(),'ref':env._sm_reference,'step':steps,'id':ids}
            for k,value in values.items():
                assert torch.isfinite(value).all(),k
                data[k].append(value.clone())
            report['real_vector_steps']=step+1
    data={k:torch.stack(v) for k,v in data.items()}
    raw=args.output_dir/'real_transition_sample.npz'
    np.savez_compressed(raw,**{k:v.cpu().numpy() for k,v in data.items()},
                        physical_joint_names=np.array(physical_names),default_q=default_q.cpu().numpy(),
                        default_qd=default_qd.cpu().numpy(),control_dt=np.array(.02))
    assert data['s'].shape==(args.steps,args.num_envs,112) and data['ref'].shape[-1]==109
    reconstructed=integrate(data['s'],delta_target(data['s'],data['s1']))
    linear=list(range(3))+list(range(7,102))+list(range(106,112))
    assert torch.allclose(reconstructed[...,linear],data['s1'][...,linear],atol=1e-5)
    for part in [slice(3,7),slice(102,106)]:
        assert ((reconstructed[...,part]*data['s1'][...,part]).sum(-1).abs()-1).abs().max()<1e-4
    wm.update_norms(data)
    losses=[]
    report.update(status='fitting_world_model',live_observation_max_error=observation_error,
                  transition_sample_sha256=sha(raw),physical_joint_names=physical_names)
    save()
    for update in range(args.model_updates):
        start=torch.randint(0,args.steps-8+1,(16,),device=args.device)
        ids=torch.randint(0,args.num_envs,(16,),device=args.device)
        times=start[:,None]+torch.arange(8,device=args.device)
        batch={k:v[times,ids[:,None]] for k,v in data.items()}
        losses.append(fit_world_model(wm,model_optimizer,batch))
        report['world_model_updates']=update+1
    starts={k:data[k][0].clone() for k in ['s','a_prev','step','id']}
    old={k:v.detach().clone() for k,v in policy.state_dict().items()}
    report.update(status='updating_policy_through_world_model');save()
    policy_metrics=improve_policy(policy,critic,wm,policy_optimizer,critic_optimizer,starts,
        env._reference_bank,default_q,default_qd,horizon=8)
    change=max(float((policy.state_dict()[k]-old[k]).abs().max()) for k in old)
    assert change>0 and policy_metrics['policy_gradient_norm']>0
    report['policy_updates']=1
    checkpoint={'policy':policy.state_dict(),'critic':critic.state_dict(),'world_model':wm.state_dict(),
        'policy_optimizer':policy_optimizer.state_dict(),'critic_optimizer':critic_optimizer.state_dict(),
        'world_model_optimizer':model_optimizer.state_dict(),'real_vector_steps':args.steps,
        'num_envs':args.num_envs,'world_model_updates':args.model_updates,'policy_updates':1,
        'default_q':default_q,'default_qd':default_qd,'physical_joint_names':physical_names,
        'asset_sha256':sha(base.ASSET),'reference_bank_sha256':env._reference_bank.manifest_sha256,
        'algorithm':'SuperMimic-style autoregressive dynamics fitting and imagined BPTT; learned task reward/termination heads; not PPO',
        'native_primitives_source':plan['existing_textop_environment']['source_root']}
    cp=args.output_dir/'integration_checkpoint.pt'
    with cp.open('xb') as f:torch.save(checkpoint,f)
    restored=torch.load(cp,map_location=args.device,weights_only=True)
    checks={k:equal_tree(checkpoint[k],restored[k]) for k in ['policy','critic','world_model',
        'policy_optimizer','critic_optimizer','world_model_optimizer']}
    assert all(checks.values())
    report.update(status='integration_smoke_completed',policy_metrics=policy_metrics,
        maximum_policy_parameter_change=change,first_world_model_losses=losses[:3],last_world_model_losses=losses[-3:],
        checkpoint_sha256=sha(cp),checkpoint_bytes=cp.stat().st_size,checkpoint_roundtrip=checks,
        actual_environment_transitions=args.steps*args.num_envs,actor_reference_input=False,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
        elapsed_seconds=time.monotonic()-began,training_budget_completed=False,task_success_claimed=False)
    save();print('SUPERMIMIC_PICK_INTEGRATION '+json.dumps(report),flush=True)
except BaseException as error:
    report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    save();print(report['traceback'],flush=True);raise
finally:
    if env is not None:env.close()
    if app is not None:app.close(wait_for_replicator=False)
