"""Classical bounded two-link positional IK, diagnostic not an innovation.

No joint-limit, torque, collision or dynamics guarantee. Two-link lengths are
preserved but this does not prove full articulated-model feasibility.
"""
import numpy as np


def knee_solution(hip, knee, ankle, delta, cap=.10):
    h,k,a,d = [np.asarray(x,dtype=np.float64) for x in (hip,knee,ankle,delta)]
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (h,k,a,d)):
        raise ValueError('Invalid leg coordinates')
    if np.linalg.norm(d)>cap+1e-12:
        raise ValueError('Foot translation exceeds bound')
    if not d.any():
        return k.copy(),np.zeros(3),1.
    l1=np.linalg.norm(k-h);l2=np.linalg.norm(a-k)
    if min(l1,l2)<1e-8:
        raise ValueError('Degenerate bone length')
    def attempt(alpha):
        target=a+alpha*d;axis=target-h;distance=np.linalg.norm(axis)
        if distance<1e-8 or distance>l1+l2+1e-12 or distance<abs(l1-l2)-1e-12:
            return None
        u=axis/distance
        along=(l1*l1-l2*l2+distance*distance)/(2*distance)
        radius=np.sqrt(max(0.,l1*l1-along*along))
        perpendicular=k-h-np.dot(k-h,u)*u
        norm=np.linalg.norm(perpendicular)
        if norm<1e-10:
            basis=np.eye(3)[np.argmin(np.abs(u))]
            perpendicular=basis-np.dot(basis,u)*u;norm=np.linalg.norm(perpendicular)
        result=h+along*u+radius*perpendicular/norm
        if np.linalg.norm(result-k)>cap+1e-12:
            return None
        return result
    result=attempt(1.)
    alpha=1.
    if result is None:
        # Conservative dyadic backoff: do not infer feasibility is monotonic.
        for exponent in range(1,41):
            alpha=2.**(-exponent);result=attempt(alpha)
            if result is not None:
                break
        if result is None:
            return k.copy(),np.zeros(3),0.
    accepted=alpha*d
    if max(abs(np.linalg.norm(result-h)-l1),abs(np.linalg.norm((a+accepted)-result)-l2))>1e-7:
        raise RuntimeError('Bone length reconstruction failed')
    return result,accepted,alpha


def numeric_checks():
    h=np.array([0.,0.,1.]);k=np.array([.15,0.,.55]);a=np.array([0.,0.,.1])
    unchanged,shift,alpha=knee_solution(h,k,a,np.zeros(3))
    if not np.array_equal(unchanged,k) or shift.any():
        raise RuntimeError('Zero-request identity failed')
    result,shift,alpha=knee_solution(h,k,a,np.array([.07,.02,0.]))
    if not shift.any() or np.linalg.norm(result-k)>.10+1e-12:
        raise RuntimeError('Reachable requested correction failed')
    if max(abs(np.linalg.norm(result-h)-np.linalg.norm(k-h)),abs(np.linalg.norm(a+shift-result)-np.linalg.norm(a-k)))>1e-7:
        raise RuntimeError('Two-link lengths differ')
    straight=np.array([0.,0.,.55])
    result,shift,alpha=knee_solution(h,straight,a,np.array([.1,0.,0.]))
    if np.linalg.norm(shift)>.10+1e-12:
        raise RuntimeError('Unreachable request escaped cap')
    try:
        knee_solution(h,k,a,np.array([.11,0.,0.]))
    except ValueError:
        pass
    else:
        raise RuntimeError('Out-of-budget shift accepted')
    return dict(status='PASS_SYNTHETIC_IK_CHECKS',tests=4,simulation_calls=0,efficacy=False)


if __name__=='__main__':
    print(numeric_checks())
