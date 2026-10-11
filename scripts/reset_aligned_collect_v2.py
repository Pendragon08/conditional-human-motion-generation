"""Measurement repair fixture; does not alter the historical experiment adapter.

The environment is reset before constructing dataset-to-world anchors. The
environment seed is fixed independently of the diffusion sampling seed.
This is a separate diagnostic, never pooled with historical outcomes.
"""
from pathlib import Path
import hashlib

base = Path(__file__).with_name('physics_guided_collect.py')
source = base.read_text()
changes = [
    ("    # Freeze one dataset-derived world coordinate frame per case and run.\n    init=",
     "    # Reset first: actor creation positions are not episode initial states.\n"
     "    reset_count=[0]\n    original_reset=task.reset\n"
     "    diffusion_calls=[0]\n    original_sample_fn=task.sample_fn\n"
     "    def counted_sample(*args,**kwargs):\n        diffusion_calls[0]+=1\n        return original_sample_fn(*args,**kwargs)\n"
     "    task.sample_fn=counted_sample\n"
     "    def observed_reset(*args,**kwargs):\n        reset_count[0]+=1\n        return original_reset(*args,**kwargs)\n"
     "    task.reset=observed_reset\n"
     "    bootstrap_obs=task._compute_observations\n"
     "    task._compute_observations=lambda *args,**kwargs: None\n"
     "    try:\n        task.reset()\n"
     "    finally:\n        task._compute_observations=bootstrap_obs\n"
     "    task._refresh_sim_tensors()\n"
     "    reset_initial_body = task._rigid_body_pos.detach().clone()\n"
     "    reset_initial_state = {key:getattr(task,key).detach().clone() for key in "
     "('_humanoid_root_states','_dof_pos','_dof_vel','_rigid_body_pos','_rigid_body_rot','_rigid_body_vel','_rigid_body_ang_vel')}\n"
     "    # Freeze the world frame after the actual episode reset.\n    init="),
    ("                  weights_frozen=True)",
     "                  weights_frozen=True, environment_seed=11,\n"
     "                  top_level_seed=int(cfg.seed), learning_seed=int(cfg.learning.params.seed),\n"
     "                  task_config_seed=int(task.cfg.get('seed',-1)),\n"
     "                  generator_config_seed=int(task.fake_mdm_args.seed),\n"
     "                  foot_asset_sha256=hashlib.sha256(Path(entry.asset_paths[id(task)]).read_bytes()).hexdigest(),\n"
     "                  initialization_fix='reset before anchoring; fixed environment seed',\n"
     "                  reset_initial_root=reset_initial_body[:,0].cpu().numpy().tolist(),\n"
     "                  reset_reference_root=dataset_sim[:,0,0].detach().cpu().numpy().tolist(),\n"
     "                  adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n"
     "                  historical_adapter_sha256=hashlib.sha256(base.read_bytes()).hexdigest())"),
    ("    def before(actions):\n        assert store['provided_frame']",
     "    def before(actions):\n"
     "        if store['steps'] == 0:\n"
     "            delta=(task._rigid_body_pos[:,0,:2]-dataset_sim[:,0,0,:2]).abs().max()\n"
     "            metadata['initial_actual_reference_xy_max_m']=float(delta)\n"
     "            metadata['actual_initial_root']=task._rigid_body_pos[:,0].detach().cpu().numpy().tolist()\n"
     "            metadata['manual_reset_vs_first_action_state_max_difference']={key:float((getattr(task,key)-value).abs().max()) for key,value in reset_initial_state.items()}\n"
     "            metadata['full_reset_calls_before_first_action']=reset_count[0]\n"
     "            metadata['upstream_warmup_diffusion_calls']=diffusion_calls[0]\n"
     "            metadata['progress_before_first_action']=task.progress_buf.detach().cpu().numpy().tolist()\n"
     "            metadata['motion_ids_before_first_action']=task._sampled_motion_ids.detach().cpu().numpy().tolist()\n"
     "            metadata['motion_start_times_before_first_action']=task._motion_start_times.detach().cpu().numpy().tolist()\n"
     "            metadata['global_offset_before_first_action']=task._global_offset.detach().cpu().numpy().tolist()\n"
     "            metadata['manual_reset_state_sha256']={key:hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for key,value in reset_initial_state.items()}\n"
     "            metadata['first_action_state_sha256']={key:hashlib.sha256(getattr(task,key).detach().cpu().numpy().tobytes()).hexdigest() for key in reset_initial_state}\n"
     "            assert float(delta)<1e-3, 'Reset/reference world anchor mismatch'\n"
     "        metadata['total_diffusion_calls_including_warmup']=diffusion_calls[0]\n"
     "        assert store['provided_frame']"),
    ("'headless=True','no_log=True','debug=False','seed='+str(a.seed),",
     "'headless=True','no_log=True','debug=False','seed=11',"),
]
for old, new in changes:
    assert source.count(old) == 1, 'Historical adapter has changed; review repair again'
    source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
