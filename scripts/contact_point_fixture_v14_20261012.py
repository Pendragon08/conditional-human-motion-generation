"""Read native contact manifolds in an unchanged known box fixture."""
from pathlib import Path

source = Path(__file__).with_name('contact_force_fixture_v3_20261012.py').read_text()
patches = [
    ('        masses = []\n', '        masses = []; envs = []\n'),
    ('            pose = gymapi.Transform(); pose.p = gymapi.Vec3(0, 0, height)\n',
     '            envs.append(env)\n            pose = gymapi.Transform(); pose.p = gymapi.Vec3(0, 0, height)\n'),
    ('        states = []; forces = []; readings = []\n',
     '        states = []; forces = []; readings = []; contact_records = []; contact_errors = []\n'),
    ('            readings.append(sensor_tensor.detach().cpu().numpy().copy())\n',
     '''            readings.append(sensor_tensor.detach().cpu().numpy().copy())
            for environment, env in enumerate(envs):
                try:
                    contacts = gym.get_env_rigid_contacts(env)
                    names = contacts.dtype.names
                    def vec(value):
                        return [float(value[key]) for key in ('x','y','z')]
                    for contact in contacts:
                        contact_records.append(dict(step=step, environment=environment,
                            body0=int(contact['body0']), body1=int(contact['body1']),
                            env0=int(contact['env0']), env1=int(contact['env1']),
                            normal=vec(contact['normal']), force_magnitude_N=float(contact['lambda']),
                            local_pos0=vec(contact['local_pos0']), local_pos1=vec(contact['local_pos1']),
                            offset0=vec(contact['offset0']), offset1=vec(contact['offset1'])))
                except Exception as error:
                    contact_errors.append(dict(step=step, environment=environment, error=repr(error)))
'''),
    ("        np.savez_compressed(out/'fixture_arrays.npz', state=states, net_contact_force=forces, constraint_force_sensor=readings)\n",
     '''        native_support = [r for r in contact_records if r['environment']==0 and r['step']>=60 and min(r['body0'],r['body1'])==-1 and r['force_magnitude_N']>0]
        native_airborne = [r for r in contact_records if r['environment']==2 and r['step']<20 and min(r['body0'],r['body1'])==-1 and r['force_magnitude_N']>0]
        checks.update(native_contact_api_available_on_stationary=bool(native_support),
                      native_airborne_has_no_ground_force=not native_airborne,
                      native_contact_api_no_python_errors=not contact_errors)
        native_status = 'AVAILABLE_ON_KNOWN_BOX' if all(checks.values()) else 'UNAVAILABLE_OR_CALIBRATION_FAILED_RETAINED'
        native = dict(status=native_status, records=contact_records, errors=contact_errors,
            sample_steps=120, environments=4, use_gpu_pipeline=False, physx_use_gpu=True,
            interpretation='Native simulated manifolds; local positions exclude thickness and offsets locate friction surface. Not real force ground truth or humanoid calibration.',
            original_physics_unchanged=True, thresholds_changed=False, innovation_confirmed=False)
        (out/'native_contacts.json').write_text(json.dumps(native,indent=2))
        np.savez_compressed(out/'fixture_arrays.npz', state=states, net_contact_force=forces, constraint_force_sensor=readings)
'''),
]
for old, new in patches:
    assert source.count(old) == 1, 'Frozen parent changed; review adapter'
    source = source.replace(old, new)
exec(compile(source, str(Path(__file__)), 'exec'), globals())
