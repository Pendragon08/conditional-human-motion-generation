"""Passive foot-sensor diagnostic; original policy inputs and metrics unchanged.

Constraint sensors are not assumed to be pure contact force on articulations.
Both net-contact and sensor readings are retained for independent calibration.
"""
from pathlib import Path

previous = Path(__file__).with_name('contact_sampling_collect_v8.py')
adapter = previous.read_text()
ending = "exec(compile(source, str(Path(__file__)), 'exec'), globals())"
assert adapter.count(ending) == 1
exec(compile(adapter.replace(ending, ''), str(previous), 'exec'), globals())

patches = [
    ("entry.official.create_rlgpu_env=create\n",
     "# Install passive instrumentation before assets are loaded. No observation\n"
     "# mode, policy, contact collection or solver setting is changed.\n"
     "from isaacgym import gymapi, gymtorch\n"
     "from closd.env.tasks.humanoid import Humanoid\n"
     "native_create_envs=Humanoid._create_envs\n"
     "sensor_bodies=['L_Ankle','L_Toe','R_Ankle','R_Toe']\n"
     "def instrumented_create_envs(self,*args,**kwargs):\n"
     "    assert self.self_obs_v == 1, 'Sensors must not change policy inputs'\n"
     "    native_load_asset=self.gym.load_asset\n"
     "    native_gym=self.gym\n"
     "    class AssetSensorProxy:\n"
     "        def __getattr__(proxy,name): return getattr(native_gym,name)\n"
     "        def load_asset(proxy,*load_args,**load_kwargs):\n"
     "            asset=native_load_asset(*load_args,**load_kwargs)\n"
     "            names=native_gym.get_asset_rigid_body_names(asset)\n"
     "            if all(name in names for name in sensor_bodies):\n"
     "                for index,name in enumerate(sensor_bodies):\n"
     "                    options=gymapi.ForceSensorProperties()\n"
     "                    options.enable_forward_dynamics_forces=False\n"
     "                    options.enable_constraint_solver_forces=True\n"
     "                    options.use_world_frame=True\n"
     "                    created=native_gym.create_asset_force_sensor(asset,names.index(name),gymapi.Transform(),options)\n"
     "                    assert created == index\n"
     "            return asset\n"
     "    self.gym=AssetSensorProxy()\n"
     "    try: return native_create_envs(self,*args,**kwargs)\n"
     "    finally: self.gym=native_gym\n"
     "Humanoid._create_envs=instrumented_create_envs\n"
     "entry.official.create_rlgpu_env=create\n"),
    ("    sim_params=task.gym.get_sim_params(task.sim)\n",
     "    constraint_sensors=gymtorch.wrap_tensor(task.gym.acquire_force_sensor_tensor(task.sim)).view(n,4,6)\n"
     "    diagnostic=[]\n"
     "    store['sensor_diagnostic']=diagnostic\n"
     "    metadata['passive_sensor_bodies']=sensor_bodies\n"
     "    metadata['passive_sensor_scope']='solver constraints only, world frame; articulated internal constraints may contribute; not ground-truth contact'\n"
     "    metadata['policy_observation_mode']=int(task.self_obs_v)\n"
     "    metadata['policy_observation_size']=int(task.num_obs)\n"
     "    metadata['sensor_metrics_used_for_selection']=False\n"
     "    sim_params=task.gym.get_sim_params(task.sim)\n"),
    ("        store['pending']={\n",
     "        diagnostic.append(dict(sensor=constraint_sensors.detach().cpu().numpy().copy(),\n"
     "            root=task._humanoid_root_states.detach().cpu().numpy().copy(),\n"
     "            dof_pos=task._dof_pos.detach().cpu().numpy().copy(),\n"
     "            dof_vel=task._dof_vel.detach().cpu().numpy().copy(),\n"
     "            action=actions.detach().cpu().numpy().copy()))\n"
     "        store['pending']={\n"),
    ("    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))\n",
     "    np.savez_compressed(out/'sensor_diagnostic.npz',**{key:np.stack([row[key] for row in store['sensor_diagnostic']]) for key in store['sensor_diagnostic'][0]})\n"
     "    metadata['sensor_diagnostic_sha256']=hashlib.sha256((out/'sensor_diagnostic.npz').read_bytes()).hexdigest()\n"
     "    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))\n"),
]
for old,new in patches:
    assert source.count(old) == 1, 'Diagnostic parent changed; re-review sensors'
    source=source.replace(old,new)
exec(compile(source,str(Path(__file__)),'exec'),globals())
