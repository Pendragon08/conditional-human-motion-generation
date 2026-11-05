"""Create isolated PyRoki helper with explicit GMR warm start; no vendor edits."""
import hashlib,json
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
src=R/'pyroki_grab_temporal_20261030/retarget_helpers/_utils.py'
out=R/'gmr_pyroki_initialization_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=src.read_text()
old='    pick_point: jnp.ndarray,\n) -> Tuple[jaxlie.SE3, jnp.ndarray]:'
new='    pick_point: jnp.ndarray,\n    gmr_root: jnp.ndarray,\n    gmr_joints: jnp.ndarray,\n) -> Tuple[jaxlie.SE3, jnp.ndarray]:'
assert s.count(old)==1
s=s.replace(old,new)
old='''    solution = (
        jaxls.LeastSquaresProblem(
            costs, [var_joints, var_Ts_world_root, var_smpl_joints_scale, var_offset]
        )
        .analyze()
        .solve()
    )'''
new='''    initial = jaxls.VarValues.make([
        var_joints.with_value(gmr_joints),
        var_Ts_world_root.with_value(jaxlie.SE3(gmr_root)),
        var_smpl_joints_scale.with_value(jnp.ones((timesteps, n_retarget, n_retarget))),
        var_offset.with_value(jnp.zeros((timesteps, 3))),
    ])
    solution = (
        jaxls.LeastSquaresProblem(
            costs, [var_joints, var_Ts_world_root, var_smpl_joints_scale, var_offset]
        )
        .analyze()
        .solve(initial_vals=initial)
    )'''
assert s.count(old)==1
s=s.replace(old,new)
out.mkdir(exist_ok=False); package=out/'retarget_helpers';package.mkdir()
(package/'__init__.py').write_text('')
target=package/'_utils.py';target.write_text(s)
with (out/'provenance.json').open('x') as f:json.dump({'parent_helper_sha256':sha(src),'helper_sha256':sha(target),'changes':'GMR root and joint states explicitly initialize JAXLS; rest of legacy objective unchanged.','training_ready':False},f,indent=2)
print(sha(target))
