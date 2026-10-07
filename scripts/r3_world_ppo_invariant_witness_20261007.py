"""CUDA-only system fixtures. No simulator rollout or scientific score."""
import argparse
from pathlib import Path
import hashlib
import json
import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from r3_world_ppo_core import HistoryBuffer,AugmentedActorCritic
from r3_world_ppo_training_core_20261007 import restore_original_adam

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True)
torch.manual_seed(20261007);device='cuda:0'
path=Path('/data/code/ouyanghao/oyh/humanoid-combat-motion/third_party/DreamControl/Training/logs/rsl_rl/g1/2026-09-27_03-19-31-Kick-a/model_1999.pt')
ckpt=torch.load(path,map_location=device,weights_only=False)
original=ActorCritic(626,626,29,actor_hidden_dims=[512,256,256],critic_hidden_dims=[512,256,256],activation='elu').to(device)
original.load_state_dict(ckpt['model_state_dict'])
policy=AugmentedActorCritic('P0',ckpt['model_state_dict']).to(device)
alg=PPO(policy,device=device,num_learning_epochs=5,num_mini_batches=4,gamma=.99,lam=.95)
restore=restore_original_adam(alg,original,ckpt);alg.init_storage('rl',4,24,[626],[626],[29])
with torch.no_grad():
    obs=torch.randn(4,626,device=device);alg.act(obs,obs);prevalue=alg.transition.values.clone()
    reward=torch.tensor([1.,2.,3.,4.],device=device)
    timeout=torch.tensor([1.,0.,0.,1.],device=device)
    done=torch.tensor([True,True,False,True],device=device)
    alg.process_env_step(reward,done,{'time_outs':timeout})
    stored=alg.storage.rewards[0,:,0].clone();expected=reward+.99*prevalue[:,0]*timeout
    error=float((stored-expected).abs().max());assert error<1e-6
    assert torch.equal(stored[1:3],reward[1:3])
    fixture=HistoryBuffer(4,device)
    raw=torch.randn(4,626,device=device)
    fixture.append(raw);first=fixture.value.clone()
    nxt=torch.randn(4,626,device=device);fixture.append(nxt)
    second=fixture.value.clone();assert torch.equal(second[:,-2],raw[:,:92])
    reset=torch.tensor([True,False,False,True],device=device)
    latest=torch.randn(4,626,device=device);fixture.append(latest,reset)
    third=fixture.value.clone()
    assert torch.count_nonzero(third[reset,:-1])==0
    assert torch.equal(third[:,-1],latest[:,:92])
    assert torch.equal(third[~reset,-3],raw[~reset,:92])
    assert torch.equal(third[~reset,-2],nxt[~reset,:92])
torch.save({'pre_action_values':prevalue,'actual_reward':reward,'timeout':timeout,'done':done,
    'stored_reward':stored,'expected_reward':expected,'history_initial_obs':raw,'history_next_obs':nxt,
    'history_last_obs':latest,'history_before_reset':second,'history_after_reset':third,'reset_mask':reset},
    a.output/'fixture_tensors.pt')
summary={'status':'COMPLETE','CUDA':True,'seed':20261007,'PPO_updates':0,'aux_updates':0,
    'simulation_steps':0,'timeout_error':error,'timeout_bootstrap_exactly_once':True,
    'true_termination_without_bootstrap':True,'history_reset_cut':True,'ordered_history_exact':True,
    'original_optimizer_restore':restore,'fixture_sha256':hashlib.sha256((a.output/'fixture_tensors.pt').read_bytes()).hexdigest(),
    'not_control_result':True}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('WORLD_PPO_INVARIANT_FIXTURE_COMPLETE',json.dumps(summary),flush=True)
