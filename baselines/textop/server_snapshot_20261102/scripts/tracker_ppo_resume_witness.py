"""Actual bounded PPO updates plus exact model/Adam/normalizer restoration.

The small diagnostic is never a40000-iteration filter run or accepted dataset.
All checkpoints/logs remain stored; shutdown starts only after writers close.
"""
import argparse,hashlib,json,pathlib,shutil,traceback
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--diagnostic-spec',required=True)
p.add_argument('--references',required=True);p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
receipt={'status':'STARTING','optimizer_updates':0,'full_training_started':False,'dataset_accepted':False,
         'physical_filter_executed':False,'output_writers_closed':False};app=None;env=None;runners=[]
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def save():(out/'receipt.json').write_text(json.dumps(receipt,indent=2))
try:
    spec=json.loads(pathlib.Path(a.spec).read_text());diag=json.loads(pathlib.Path(a.diagnostic_spec).read_text())
    assert a.headless and a.device==diag['device']=='cuda:1';assert shutil.disk_usage(out).free>1024**3
    references=pathlib.Path(a.references);meta=json.loads((references/'receipt.json').read_text())
    assert meta['status']=='TEXTOP_TRACKER_REFERENCE_EXPORT_PASSED'
    assert sha(references/'manifest.private.jsonl')==meta['private_manifest_sha256']
    rows=[json.loads(x) for x in (references/'manifest.private.jsonl').read_text().splitlines()]
    assert rows and all(sha(x['file'])==x['sha256'] for x in rows)
    receipt.update(source_sha256=sha(__file__),diagnostic_spec_sha256=sha(a.diagnostic_spec),reference_trajectories=len(rows))
    save();app=AppLauncher(a).app
    import torch,gymnasium as gym
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from paper_filter_config import make,paper_settings
    torch.set_num_threads(4);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
    torch.cuda.set_per_process_memory_fraction(.65,torch.device(a.device))
    cfg,agent=make(spec,[x['file'] for x in rows],out/'usd',phase='relaxed',num_envs=diag['num_envs'])
    env=RslRlVecEnvWrapper(gym.make('Tracking-Flat-G1-PrivPrivObs-v0',cfg=cfg))
    assert env.get_observations()[0].shape==(8,557)
    count=[0]
    def attach_counter(runner):
        original=runner.alg.optimizer.step
        def step(*args,**kwargs):
            assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in runner.alg.policy.parameters())
            result=original(*args,**kwargs);count[0]+=1;receipt['optimizer_updates']=count[0];return result
        runner.alg.optimizer.step=step
    def equal(x,y):
        if torch.is_tensor(x):return x.dtype==y.dtype and x.shape==y.shape and torch.equal(x.detach().cpu(),y.detach().cpu())
        if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
        if isinstance(x,(tuple,list)):return len(x)==len(y) and all(equal(u,v) for u,v in zip(x,y))
        return x==y
    def close_writer(runner):
        if runner.writer is not None:runner.writer.flush();runner.writer.close();runner.writer=None
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(out/'before_resume'),device=a.device);runners.append(runner)
    before=[p.detach().clone() for p in runner.alg.policy.parameters()];attach_counter(runner)
    runner.learn(num_learning_iterations=1,init_at_random_ep_len=False)
    assert count[0]==20
    delta=max(float((x-y).abs().max()) for x,y in zip(before,runner.alg.policy.parameters()));assert delta>0
    base_counter=runner.current_learning_iteration
    # RSL-RL serializes the last loop index. Explicitly save the completed count
    # in this separate resume checkpoint so iteration0 is not repeated onresume.
    runner.current_learning_iteration=1
    checkpoint=out/'completed_iteration_1.pt';runner.save(str(checkpoint),infos={'completed_iterations':1,'actual_optimizer_updates':20})
    state=torch.load(checkpoint,map_location=a.device,weights_only=False);close_writer(runner)
    _,fresh_agent=make(spec,[x['file'] for x in rows],out/'usd',phase='relaxed',num_envs=8)
    restored=OnPolicyRunner(env,fresh_agent.to_dict(),log_dir=str(out/'after_resume'),device=a.device);runners.append(restored)
    infos=restored.load(str(checkpoint),load_optimizer=True)
    assert infos['completed_iterations']==1 and restored.current_learning_iteration==1
    assert equal(restored.alg.policy.state_dict(),state['model_state_dict'])
    assert equal(restored.alg.optimizer.state_dict(),state['optimizer_state_dict'])
    assert equal(restored.obs_normalizer.state_dict(),state['obs_norm_state_dict'])
    assert equal(restored.privileged_obs_normalizer.state_dict(),state['privileged_obs_norm_state_dict'])
    restored_before=[p.detach().clone() for p in restored.alg.policy.parameters()];attach_counter(restored)
    restored.learn(num_learning_iterations=1,init_at_random_ep_len=False)
    assert count[0]==diag['expected_actual_optimizer_updates']==40
    resumed_delta=max(float((x-y).abs().max()) for x,y in zip(restored_before,restored.alg.policy.parameters()));assert resumed_delta>0
    restored.current_learning_iteration=2;final=out/'completed_iteration_2.pt'
    restored.save(str(final),infos={'completed_iterations':2,'actual_optimizer_updates':40});close_writer(restored)
    receipt.update(status='TEXTOP_TRACKER_PPO_RESUME_WITNESS_PASSED',optimizer_updates=count[0],num_envs=8,
        completed_learning_iterations=2,environment_transitions=8*24*2,policy_observation_dim=557,
        model_restored_exactly=True,adam_restored_exactly=True,both_normalizers_restored_exactly=True,
        restore_comparison='exact tensor values/dtypes/shapes; Adam scalar step CPU residency is permitted by PyTorch',
        initial_parameter_max_change=delta,resumed_parameter_max_change=resumed_delta,
        underlying_first_last_index=base_counter,resume_completed_iteration=1,
        checkpoints=[{'name':x.name,'bytes':x.stat().st_size,'sha256':sha(x)} for x in [checkpoint,final]],
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(a.device)),paper_settings=paper_settings(),
        output_writers_closed=True,full_filter_training_ready=False)
    save();print('TEXTOP_TRACKER_PPO_RESUME_WITNESS_PASSED',flush=True)
except BaseException as error:
    receipt.update(status='FAILED',error=repr(error),traceback=traceback.format_exc());save();raise
finally:
    for runner in runners:
        if runner.writer is not None:runner.writer.flush();runner.writer.close()
    if app is not None:
        from bounded_headless_close import close
        close(app,receipt,save,out,simulation=env.unwrapped.sim if env is not None else None,
              successful_statuses=('TEXTOP_TRACKER_PPO_RESUME_WITNESS_PASSED',))
