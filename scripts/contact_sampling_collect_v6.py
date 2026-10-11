"""Completion-fence diagnostic. Historical adapters remain byte-identical."""
from pathlib import Path

previous = Path(__file__).with_name('contact_sampling_collect_v5.py')
adapter = previous.read_text()
ending = "exec(compile(source, str(Path(__file__)), 'exec'), globals())"
assert adapter.count(ending) == 1
adapter = adapter.replace(ending, '')
exec(compile(adapter, str(previous), 'exec'), globals())

patches = [
    ("    reset_count=[0]\n    original_reset=task.reset\n",
     "    reset_count=[0]\n    original_reset=task.reset\n"
     "    completion_fences=[0]\n"
     "    original_physics_step=task._physics_step\n"
     "    def completed_physics_step(*args,**kwargs):\n"
     "        result=original_physics_step(*args,**kwargs)\n"
     "        task.gym.fetch_results(task.sim,True)\n"
     "        completion_fences[0]+=1\n"
     "        return result\n"
     "    task._physics_step=completed_physics_step\n"),
    ("        return original_reset(*args,**kwargs)\n",
     "        result=original_reset(*args,**kwargs)\n"
     "        task.gym.fetch_results(task.sim,True)\n"
     "        completion_fences[0]+=1\n"
     "        return result\n"),
    ("    store.update(metadata=metadata,steps=0,",
     "    sim_params=task.gym.get_sim_params(task.sim)\n"
     "    metadata['completion_fence_protocol']='fetch_results(True) after native physics step and native reset; no extra simulate or tensor refresh'\n"
     "    metadata['sim_params']=dict(dt=float(sim_params.dt),substeps=int(sim_params.substeps),use_gpu_pipeline=bool(sim_params.use_gpu_pipeline),contact_collection=int(sim_params.physx.contact_collection),physx_use_gpu=bool(sim_params.physx.use_gpu))\n"
     "    metadata['contact_mutation_during_reward_max_N']=0.0\n"
     "    store.update(metadata=metadata,steps=0,"),
    ("    def record(actions):\n        result=reward(actions)\n",
     "    def record(actions):\n"
     "        torch.cuda.synchronize()\n"
     "        force_before_reward=task._contact_forces.detach().clone()\n"
     "        result=reward(actions)\n"),
    ("        store['pending']={\n",
     "        metadata['contact_mutation_during_reward_max_N']=max(metadata['contact_mutation_during_reward_max_N'],float((task._contact_forces-force_before_reward).abs().max()))\n"
     "        metadata['completion_fence_calls']=completion_fences[0]\n"
     "        store['pending']={\n"),
]
for old, new in patches:
    assert source.count(old) == 1, 'Upstream diagnostic changed; re-review fence'
    source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
