"""Necessary vertical-force bound under explicit ground-only assumptions.

If gravity is the only downward external force and contacts push upward, the
mass-normalized vertical contact force cannot be negative. This is a necessary
condition on a sampled model trajectory, not evidence of actual contact or a
sufficient dynamical feasibility test.
"""
import numpy as np


def vertical_support_bound(times, com_xyz, gravity_xyz):
    t=np.asarray(times,dtype=np.float64); c=np.asarray(com_xyz,dtype=np.float64); g=np.asarray(gravity_xyz,dtype=np.float64)
    if t.ndim!=1 or len(t)<3 or c.shape!=(len(t),3) or g.shape!=(3,) or not all(np.isfinite(a).all() for a in (t,c,g)):
        raise ValueError('finite time, COM and gravity arrays required')
    dt=np.diff(t)
    if np.any(dt<=0) or not np.allclose(dt,dt[0],rtol=0,atol=1e-9):raise ValueError('strictly uniform increasing times required')
    if np.linalg.norm(g[:2])>1e-10 or g[2]>=0:raise ValueError('gravity must be downward along the z axis')
    accel=(c[2:]-2*c[1:-1]+c[:-2])/(dt[0]**2)
    vertical=accel[:,2]-g[2]
    return {'center_times':t[1:-1],'com_acceleration_m_s2':accel,
            'required_vertical_support_n_per_kg':vertical,
            'negative_vertical_support_samples':int((vertical<0).sum()),
            'training_ready':False}
