"""Bounded experimental root-height adjustment for a necessary support bound.

This is not a DreamControl-v2 method, contact model or physical feasibility
filter. Caller must choose and document bounds, gravity and interpretation.
"""
import numpy as np
from scipy.optimize import LinearConstraint, Bounds, minimize
from vertical_support_bound import vertical_support_bound


def project_vertical_support(times, com_xyz, gravity_xyz, *, max_shift_m, smoothness_weight,
                             numerical_tolerance=1e-6):
    if (isinstance(max_shift_m,bool) or not np.isfinite(max_shift_m) or max_shift_m<=0 or
        isinstance(smoothness_weight,bool) or not np.isfinite(smoothness_weight) or smoothness_weight<0 or
        not np.isfinite(numerical_tolerance) or numerical_tolerance<=0):raise ValueError('invalid explicit projection parameters')
    t=np.asarray(times,dtype=np.float64);com=np.asarray(com_xyz,dtype=np.float64)
    before=vertical_support_bound(t,com,gravity_xyz)['required_vertical_support_n_per_kg']
    n=len(t);dt=t[1]-t[0]
    d2=np.zeros((n-2,n));i=np.arange(n-2);d2[i,i]=1;d2[i,i+1]=-2;d2[i,i+2]=1
    d1=np.zeros((n-1,n));j=np.arange(n-1);d1[j,j]=-1;d1[j,j+1]=1
    h=np.eye(n)+smoothness_weight*(d1.T@d1)
    def fun(x):return .5*float(x@h@x)
    def jac(x):return h@x
    lo=np.full(n,-max_shift_m);hi=np.full(n,max_shift_m);lo[[0,-1]]=0;hi[[0,-1]]=0
    constraint=LinearConstraint(d2,-before*dt**2,np.full(n-2,np.inf))
    result=minimize(fun,np.zeros(n),jac=jac,method='SLSQP',bounds=Bounds(lo,hi),constraints=[constraint],
                    options={'ftol':1e-12,'maxiter':1000,'disp':False})
    if not result.success or not np.isfinite(result.x).all():raise ValueError('bounded support projection failed: '+str(result.message))
    delta=result.x
    adjusted=com.copy();adjusted[:,2]+=delta
    after=vertical_support_bound(t,adjusted,gravity_xyz)['required_vertical_support_n_per_kg']
    if (after.min() < -numerical_tolerance or np.max(np.abs(delta))>max_shift_m+1e-10 or
        abs(delta[0])>1e-10 or abs(delta[-1])>1e-10):raise ValueError('projection did not satisfy its numerical constraints')
    return {'height_shift_m':delta,'before_n_per_kg':before,'after_n_per_kg':after,
            'solver_iterations':int(result.nit),'training_ready':False}
