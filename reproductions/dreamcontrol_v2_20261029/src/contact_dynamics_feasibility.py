"""Experimental sampled-state rigid-body force-balance *necessary* check.

This is not a DreamControl-v2 method, a controller, a contact detector, or an
overall quality gate. Inputs must come from separately verified model/state,
contact geometry, time and units. Each point Jacobian must be expressed in an
orthonormal contact basis (tangent 1, tangent 2, outward normal), with the
normal force positive on the robot. The inscribed friction pyramid is more
restrictive than circular Coulomb friction; failure does not establish physical
impossibility. A solution omits contact acceleration/complementarity, collision,
inter-frame dynamics, motor curves and model uncertainty.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog


def check_force_balance(mass, acceleration, bias, actuation, contact_jacobians, *,
                        torque_lower, torque_upper, friction, tolerance):
    """Test M*a+bias = actuation*tau + sum(J_i.T*f_i) with point forces.

    `contact_jacobians` has shape (K, 3, D). Friction uses the conservative
    |ft1|+|ft2| <= mu*fn cone, not a paper-specified threshold. Returns solver
    evidence, never a dataset acceptance or `training_ready=True`.
    """
    mass=np.asarray(mass,dtype=float); acceleration=np.asarray(acceleration,dtype=float)
    bias=np.asarray(bias,dtype=float); actuation=np.asarray(actuation,dtype=float)
    jac=np.asarray(contact_jacobians,dtype=float); lo=np.asarray(torque_lower,dtype=float)
    hi=np.asarray(torque_upper,dtype=float); mu=np.asarray(friction,dtype=float)
    if acceleration.ndim!=1 or not len(acceleration):raise ValueError('nonempty acceleration vector required')
    d=len(acceleration)
    if (mass.shape!=(d,d) or bias.shape!=(d,) or actuation.ndim!=2 or actuation.shape[0]!=d or
        jac.ndim!=3 or jac.shape[1:]!=(3,d) or lo.shape!=(actuation.shape[1],) or
        hi.shape!=lo.shape or mu.shape!=(jac.shape[0],)):
        raise ValueError('incompatible dynamics dimensions')
    if (isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or
        not np.isfinite(tolerance) or tolerance<=0 or
        not all(np.isfinite(x).all() for x in (mass,acceleration,bias,actuation,jac,lo,hi,mu)) or
        np.any(lo>hi) or np.any(mu<0)):
        raise ValueError('invalid finite dynamics inputs or bounds')
    if not np.allclose(mass,mass.T,rtol=0,atol=1e-10):raise ValueError('mass matrix must be symmetric')
    try:np.linalg.cholesky(mass)
    except np.linalg.LinAlgError:raise ValueError('mass matrix must be positive definite') from None
    n=actuation.shape[1];k=jac.shape[0]
    equality=np.concatenate([actuation,jac.transpose(2,0,1).reshape(d,3*k)],axis=1)
    rhs=mass@acceleration+bias
    if n+3*k==0:
        residual=float(np.max(np.abs(rhs)))
        return {'status':'feasible' if residual<=tolerance else 'infeasible_no_forces',
                'max_equilibrium_residual':residual,'torques':[],'contact_forces':[],
                'training_ready':False}
    inequalities=[]
    for i in range(k):
        for s1,s2 in ((1,1),(1,-1),(-1,1),(-1,-1)):
            row=np.zeros(n+3*k);row[n+3*i:n+3*i+3]=(s1,s2,-mu[i]);inequalities.append(row)
    bounds=[(float(a),float(b)) for a,b in zip(lo,hi)]
    for _ in range(k):bounds.extend([(None,None),(None,None),(0,None)])
    result=linprog(np.zeros(n+3*k),A_ub=np.array(inequalities) if inequalities else None,
                   b_ub=np.zeros(len(inequalities)) if inequalities else None,
                   A_eq=equality,b_eq=rhs,bounds=bounds,method='highs')
    if not result.success:
        status=('infeasible_inner_friction_pyramid' if k else 'infeasible_force_balance') if result.status==2 else 'solver_indeterminate'
        return {'status':status,
                'solver_status':int(result.status),'solver_message':str(result.message),'training_ready':False}
    if result.x is None or not np.isfinite(result.x).all():
        return {'status':'solver_indeterminate','solver_status':int(result.status),
                'solver_message':'solver returned non-finite variables','training_ready':False}
    forces=result.x[n:].reshape(k,3);torques=result.x[:n]
    residual=float(np.max(np.abs(equality@result.x-rhs)))
    violation=max([0.]+[float(abs(f[0])+abs(f[1])-m*f[2]) for f,m in zip(forces,mu)] +
                  [float(-f[2]) for f in forces] +
                  [float(l-v) for l,v in zip(lo,torques)] + [float(v-h) for v,h in zip(torques,hi)])
    valid=residual<=tolerance and violation<=tolerance
    return {'status':'feasible' if valid else 'numerical_residual_exceeded',
            'max_equilibrium_residual':residual,'max_bound_or_cone_violation':violation,
            'torques':torques.tolist(),'contact_forces':forces.tolist(),
            'solver_status':int(result.status),'training_ready':False,
            'limitations':['Conditional on unverified supplied model, acceleration, time, contact and friction.',
                           'Inscribed friction pyramid; infeasibility does not exclude circular-cone solutions.',
                           'No contact kinematics, complementarity, collision, actuator curves or trajectory verification.']}
