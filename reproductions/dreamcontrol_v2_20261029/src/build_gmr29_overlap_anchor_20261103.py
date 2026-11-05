"""Build project-local PyRoki helper with a soft previous-window anchor."""
import hashlib,json
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
src=R/'gmr_pyroki_initialization_20261103/retarget_helpers/_utils.py';out=R/'gmr29_overlap_anchor_helper_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=src.read_text();old='    gmr_joints: jnp.ndarray,\n) -> Tuple[jaxlie.SE3, jnp.ndarray]:';new='    gmr_joints: jnp.ndarray,\n    anchor_root: jnp.ndarray,\n    anchor_joints: jnp.ndarray,\n    anchor_mask: jnp.ndarray,\n) -> Tuple[jaxlie.SE3, jnp.ndarray]:'
if s.count(old)!=1:raise ValueError('signature changed')
s=s.replace(old,new)
old='''    costs = [
        # Costs that are relatively self-contained to the robot.'''
new='''    @jaxls.Cost.create_factory
    def prior_overlap_cost(var_values, var_Ts_world_root, var_robot_cfg,
                           var_offset, target_root, target_joints, mask):
        actual = jaxlie.SE3.from_translation(var_values[var_offset]) @ var_values[var_Ts_world_root]
        target = jaxlie.SE3(target_root)
        relative = (target.inverse() @ actual).log()
        joints = var_values[var_robot_cfg]
        return jnp.concatenate([(joints - target_joints) * 10.0,
                                relative[:3] * 10.0, relative[3:] * 10.0]) * mask

    costs = [
        prior_overlap_cost(var_Ts_world_root, var_joints, var_offset,
                           anchor_root, anchor_joints, anchor_mask),
        # Costs that are relatively self-contained to the robot.'''
if s.count(old)!=1:raise ValueError('cost insertion point changed')
s=s.replace(old,new)
out.mkdir(exist_ok=False);d=out/'retarget_helpers';d.mkdir();(d/'__init__.py').write_text('');p=d/'_utils.py';p.write_text(s)
with (out/'provenance.json').open('x') as f:json.dump({'source_helper_sha256':sha(src),'helper_sha256':sha(p),'modification':'soft overlap anchor, weight 10 for 29 joints and SE3 logarithm; caller-provided 0/1 mask','training_ready':False},f,indent=2)
print(sha(p))
if __name__=='__main__':pass
