"""Small synthetic position/orientation IK on G1, CPU only; not retargeting."""
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import jaxlie
import jaxls
import pyroki as pk
import yourdfpy


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    if any(x.platform!='cpu' for x in jax.devices()): raise ValueError('CPU required')
    path=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    robot=pk.Robot.from_urdf(yourdfpy.URDF.load(str(path),load_meshes=False))
    name='right_wrist_yaw_link'; link=robot.links.names.index(name)
    q0=np.asarray(robot.joint_var_cls.default_factory()).copy()
    lo=np.asarray(robot.joints.lower_limits); hi=np.asarray(robot.joints.upper_limits)
    target_q=q0.copy(); index=robot.joints.actuated_names.index('right_elbow_joint')
    target_q[index]=np.clip(target_q[index]+.25,lo[index]+.01,hi[index]-.01)
    target_pose=np.asarray(robot.forward_kinematics(jnp.asarray(target_q)))[link]
    base_pose=np.asarray(robot.forward_kinematics(jnp.asarray(q0)))[link]
    initial=float(np.linalg.norm(base_pose[4:7]-target_pose[4:7]))
    if initial<1e-4: raise ValueError('synthetic target is not distinct')
    var=robot.joint_var_cls(0)
    costs=[pk.costs.pose_cost_analytic_jac(robot,var,jaxlie.SE3(jnp.asarray(target_pose)),jnp.array(link),pos_weight=50.,ori_weight=10.),
           pk.costs.limit_constraint(robot,var)]
    sol=(jaxls.LeastSquaresProblem(costs=costs,variables=[var]).analyze().solve(
        verbose=False,linear_solver='dense_cholesky',trust_region=jaxls.TrustRegionConfig(lambda_initial=1.)))
    fitted=np.asarray(sol[var]); actual=np.asarray(robot.forward_kinematics(jnp.asarray(fitted)))[link]
    final=float(np.linalg.norm(actual[4:7]-target_pose[4:7]))
    if not np.isfinite(fitted).all() or final >= initial or np.any(fitted<lo-1e-3) or np.any(fitted>hi+1e-3):
        raise ValueError('synthetic IK did not improve within joint limits')
    report={'jax_version':jax.__version__,'devices':[str(x) for x in jax.devices()],
            'urdf_sha256':sha(path),'script_sha256':sha(__file__),'link':name,
            'initial_position_error_m':initial,'final_position_error_m':final,
            'max_joint_limit_excess_rad':float(max(0.,np.max(lo-fitted),np.max(fitted-hi))),
            'warning':'Single synthetic wrist pose IK with fixed root; no human data, 12-point retargeting, contacts, dynamics or reproduction validation.'}
    with Path(__file__).with_name('ik_report.json').open('x',encoding='utf8') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
