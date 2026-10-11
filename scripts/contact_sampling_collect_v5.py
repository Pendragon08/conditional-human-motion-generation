"""Synchronized sampling diagnostic adapter v5; does not alter the historical experiment adapter.

The environment is reset before constructing dataset-to-world anchors. The
environment seed is fixed independently of the diffusion sampling seed.
This is a separate diagnostic, never pooled with historical outcomes.
"""
from pathlib import Path
import hashlib

base = Path(__file__).with_name('physics_guided_collect.py')
source = base.read_text()
changes = [
    ("    assert dataset_sim.shape==(n,120,24,3)\n",
     "    # Define a source-to-world translation before freezing the original target.\n"
     "    bootstrap_reference_root_xy_delta=reset_initial_body[:,0,:2]-dataset_sim[:,0,0,:2]\n"
     "    before_horizontal_alignment=dataset_sim.detach().clone()\n"
     "    alignment_isaac_delta=torch.cat([bootstrap_reference_root_xy_delta,torch.zeros((n,1),device=task.device)],dim=1)\n"
     "    dataset_pose+=alignment_isaac_delta.matmul(task.to_isaac_mat)[:,None,None,:]\n"
     "    dataset_sim=dataset_pose[:,:,smpl_2_mujoco].matmul(task.to_isaac_mat.T)\n"
     "    alignment_relative_pose_error=float(((dataset_sim-dataset_sim[:,:,0:1])-(before_horizontal_alignment-before_horizontal_alignment[:,:,0:1])).abs().max())\n"
     "    alignment_displacement_error=float(((dataset_sim[:,1:]-dataset_sim[:,:-1])-(before_horizontal_alignment[:,1:]-before_horizontal_alignment[:,:-1])).abs().max())\n"
     "    alignment_z_error=float((dataset_sim[:,:,:,2]-before_horizontal_alignment[:,:,:,2]).abs().max())\n"
     "    assert max(alignment_relative_pose_error,alignment_displacement_error,alignment_z_error)<1e-5\n"
     "    assert dataset_sim.shape==(n,120,24,3)\n"),
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
     "    # Do not refresh here: official reset has already restored its one-shot rigid-body cache.\n"
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
     "                  initialization_fix='reset before anchoring; fixed environment seed; no redundant rigid-body refresh',\n"
     "                  reset_initial_root=reset_initial_body[:,0].cpu().numpy().tolist(),\n"
     "                  reset_reference_root=dataset_sim[:,0,0].detach().cpu().numpy().tolist(),\n"
     "                  adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n"
     "                  reference_horizontal_alignment_delta_m=bootstrap_reference_root_xy_delta.detach().cpu().numpy().tolist(),\n"
     "                  reference_alignment_invariants=dict(relative_pose_max_error_m=alignment_relative_pose_error,displacement_max_error_m=alignment_displacement_error,z_max_error_m=alignment_z_error),\n"
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
     "            snapshots={('bootstrap_'+key):value.cpu().numpy() for key,value in reset_initial_state.items()}\n"
     "            snapshots.update({('first_action_'+key):getattr(task,key).detach().cpu().numpy() for key in reset_initial_state})\n"
     "            snapshots['first_action_contact_forces']=task._contact_forces.detach().cpu().numpy()\n"
     "            np.savez_compressed(out/'initial_states.npz',**snapshots)\n"
     "            metadata['initial_state_archive_sha256']=hashlib.sha256((out/'initial_states.npz').read_bytes()).hexdigest()\n"
     "            metadata['initial_rigid_cache_consistency']={key:float((getattr(task,key)-getattr(task,cache_key)).abs().max()) for key,cache_key in [('_rigid_body_pos','_reset_rb_pos'),('_rigid_body_rot','_reset_rb_rot'),('_rigid_body_vel','_reset_rb_vel'),('_rigid_body_ang_vel','_reset_rb_ang_vel')]}\n"
     "            metadata['initial_reset_gate_pass']=all(value==0 for value in metadata['manual_reset_vs_first_action_state_max_difference'].values()) and all(value==0 for value in metadata['initial_rigid_cache_consistency'].values())\n"
     "            if not metadata['initial_reset_gate_pass']:\n                (out/'initialization_failure.json').write_text(json.dumps(metadata,indent=2))\n"
     "            assert metadata['initial_reset_gate_pass'], 'Full exported reset state/cache differs before first action'\n"
     "            metadata['first_action_state_sha256']={key:hashlib.sha256(getattr(task,key).detach().cpu().numpy().tobytes()).hexdigest() for key in reset_initial_state}\n"
     "            if float(delta)>=1e-3:\n                (out/'initialization_failure.json').write_text(json.dumps(metadata,indent=2))\n            assert float(delta)<1e-3, 'Reset/reference world anchor mismatch'\n"
     "        metadata['total_diffusion_calls_including_warmup']=diffusion_calls[0]\n"
     "        assert store['provided_frame']"),
    ("'headless=True','no_log=True','debug=False','seed='+str(a.seed),",
     "'headless=True','no_log=True','debug=False','seed=11',"),
]
changes.append((
    "    def record(actions):\n        result=reward(actions)\n",
    "    def record(actions):\n        result=reward(actions)\n        torch.cuda.synchronize()  # Measure all producer streams after Physics refresh.\n"))
for old, new in changes:
    assert source.count(old) == 1, 'Historical adapter has changed; review repair again'
    source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
