"""CPU-only PyRoki 27DoF G1 FK and derivative check against separate URDF FK."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
import pyroki
import yourdfpy

ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
sys.path.insert(0,str(ROOT/'foot_contact_fit_20261030'))
from g1_fk import URDFFK
from waist_lock_ik import FIT_LINKS


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    if any(x.platform!='cpu' for x in jax.devices()): raise ValueError('CPU required')
    urdf_path=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf')
    urdf=yourdfpy.URDF.load(str(urdf_path),load_meshes=False)
    robot=pyroki.Robot.from_urdf(urdf)
    names=list(robot.joints.actuated_names)
    reference=URDFFK(urdf_path)
    if len(names)!=27 or len(set(names))!=27 or set(names)!=set(reference.actuated):
        raise ValueError('joint mapping mismatch')
    indices=[robot.links.names.index(x) for x in FIT_LINKS]
    rng=np.random.default_rng(20261030)
    lower=np.asarray(robot.joints.lower_limits); upper=np.asarray(robot.joints.upper_limits)
    q=lower[None,:]+(.2+.6*rng.random((8,27)))*(upper-lower)[None,:]
    if not np.isfinite(q).all(): raise ValueError('non-finite limits')
    actual=np.asarray(robot.forward_kinematics(jnp.asarray(q)))[:,indices,4:7]
    expected=reference.forward(np.zeros((8,3)),np.tile([0.,0.,0.,1.],(8,1)),q,names,links=FIT_LINKS)
    err=np.linalg.norm(actual-expected,axis=-1)
    grad=np.asarray(jax.jacfwd(lambda x:robot.forward_kinematics(x)[indices,4:7])(jnp.asarray(q[0])))
    if not np.isfinite(grad).all() or grad.shape!=(12,3,27): raise ValueError('invalid FK Jacobian')
    if float(err.max()) > 1e-5: raise ValueError('FK mismatch exceeds tolerance')
    report={'jax_version':jax.__version__,'devices':[str(x) for x in jax.devices()],
            'urdf_sha256':sha(urdf_path),'script_sha256':sha(__file__),'joint_names':names,
            'links':list(FIT_LINKS),'samples':len(q),'position_max_error_m':float(err.max()),
            'position_mean_error_m':float(err.mean()),'jacobian_shape':list(grad.shape),
            'jacobian_finite':True,'jacobian_max_abs':float(np.abs(grad).max()),
            'warning':'FK and autodiff only; neither retarget optimizer, contact, human data nor training acceptance.'}
    with Path(__file__).with_name('fk_report.json').open('x',encoding='utf8') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
