"""Analytical fixtures for the surface proxy; it is not contact-point truth."""
import numpy as np
from physics_guided_core import surface_metrics


def main():
    names=['L_Ankle','L_Toe','R_Ankle','R_Toe']
    geometry={name:dict(points=[[.1,0,-.1]],radii=[0.]) for name in names}
    thresholds=dict(contact_force_n=10.,surface_near_ground_m=.02,
        sliding_speed_m_s=.05,penetration_depth_m=.01)
    pos=np.zeros((10,4,3)); pos[:,:,2]=.1
    quat=np.zeros((10,4,4)); quat[:,:,3]=1.
    lin=np.zeros_like(pos); ang=np.zeros_like(pos); force=np.zeros_like(pos); force[:,:,2]=100.
    fixed=surface_metrics(pos,quat,lin,ang,force,names,geometry,thresholds)
    assert fixed['surface_slip_proxy_fraction']==0 and fixed['penetration_fraction']==0
    # Moving body center can have stationary contact material point when rotating.
    lin[:,:,0]=.1; ang[:,:,1]=1.
    rolling=surface_metrics(pos,quat,lin,ang,force,names,geometry,thresholds)
    assert rolling['surface_slip_proxy_fraction']==0
    ang[:]=0
    slipping=surface_metrics(pos,quat,lin,ang,force,names,geometry,thresholds)
    assert slipping['surface_slip_proxy_fraction']==1
    pos[:,:,2]=.5; force[:]=0
    airborne=surface_metrics(pos,quat,lin,ang,force,names,geometry,thresholds)
    assert airborne['surface_slip_proxy_fraction'] is None and airborne['airborne_fraction']==1
    assert airborne['penetration_fraction']==0
    pos[:,:,2]=.08; force[:,:,2]=100.
    penetrated=surface_metrics(pos,quat,lin,ang,force,names,geometry,thresholds)
    assert penetrated['penetration_fraction']==1
    print('PHYSICS_GUIDED_ANALYTIC_SURFACE_CHECK_PASS')


if __name__=='__main__':
    main()
