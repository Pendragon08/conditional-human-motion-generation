"""Read-only native manifold exporter; no policy or physics mutation."""
import numpy as np


def field(record, logical):
    matches = [name for name in record.dtype.names
               if name.replace('_', '').lower() == logical.replace('_', '').lower()]
    assert len(matches) == 1, (logical, record.dtype.names)
    return record[matches[0]]


def vector(value):
    return [float(value[name]) for name in ('x', 'y', 'z')]


def read_contacts(gym, envs, step, records, errors, dtypes):
    for environment, env in enumerate(envs):
        try:
            contacts = gym.get_env_rigid_contacts(env)
            names = list(contacts.dtype.names)
            if names not in dtypes:
                dtypes.append(names)
            for contact in contacts:
                row = dict(step=step, environment=environment)
                for name in ('body0', 'body1', 'env0', 'env1'):
                    row[name] = int(field(contact, name))
                for name in ('normal', 'local_pos0', 'local_pos1', 'offset0', 'offset1'):
                    row[name] = vector(field(contact, name))
                row['force_magnitude_N'] = float(field(contact, 'lambda'))
                assert np.isfinite([row['force_magnitude_N']] + sum(
                    [row[name] for name in ('normal', 'local_pos0', 'local_pos1', 'offset0', 'offset1')], [])).all()
                records.append(row)
        except Exception as error:
            errors.append(dict(step=step, environment=environment, error=repr(error)))


def body_properties(gym, envs, handles):
    return [[dict(mass=float(prop.mass), com=[float(prop.com.x), float(prop.com.y), float(prop.com.z)])
             for prop in gym.get_actor_rigid_body_properties(env, handle)]
            for env, handle in zip(envs, handles)]
