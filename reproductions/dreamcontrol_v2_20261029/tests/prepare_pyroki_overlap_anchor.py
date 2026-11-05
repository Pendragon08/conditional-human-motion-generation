"""Create experimental soft overlap anchor from the isolated temporal helper."""
import hashlib,json
from pathlib import Path
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029')
HERE=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def replace_once(s,a,b):
    if s.count(a)!=1:raise ValueError('expected one match: '+a)
    return s.replace(a,b)
def save(p,s):
    with p.open('x',encoding='utf8',newline='\n') as f:f.write(s)

def main():
    base=ROOT/'pyroki_grab_temporal_20261030';receipt=json.loads((base/'report.json').read_text())
    source=base/'retarget_helpers/_utils.py'
    if sha(source)!=receipt['helper_sha256']:raise ValueError('unexpected helper')
    s=source.read_text()
    s=replace_once(s,'    pick_point: jnp.ndarray,\n) -> Tuple[jaxlie.SE3, jnp.ndarray]:',
       '    pick_point: jnp.ndarray,\n    anchor_roots: jnp.ndarray,\n    anchor_joints: jnp.ndarray,\n    anchor_mask: jnp.ndarray,\n    anchor_weight: float,\n) -> Tuple[jaxlie.SE3, jnp.ndarray]:')
    marker='    costs = [\n        # Costs that are relatively self-contained to the robot.'
    insert='''    @jaxls.Cost.create_factory
    def overlap_anchor_cost(var_values: jaxls.VarValues, root_var: jaxls.SE3Var,
                            joint_var: jaxls.Var[jnp.ndarray], offset_var: OffsetVar,
                            target_root: jnp.ndarray, target_joints: jnp.ndarray,
                            active: jnp.ndarray) -> jax.Array:
        current = jaxlie.SE3.from_translation(var_values[offset_var]) @ var_values[root_var]
        delta = (jaxlie.SE3(target_root).inverse() @ current).log()
        return jnp.concatenate([delta.flatten(), (var_values[joint_var]-target_joints).flatten()]) * active * anchor_weight

    costs = [
        overlap_anchor_cost(var_Ts_world_root, var_joints, var_offset, anchor_roots, anchor_joints, anchor_mask),
        # Costs that are relatively self-contained to the robot.'''
    s=replace_once(s,marker,insert)
    d=HERE/'retarget_helpers';d.mkdir(exist_ok=False);save(d/'__init__.py','');save(d/'_utils.py',s)
    doc={'baseline_helper_sha256':sha(source),'anchor_helper_sha256':sha(d/'_utils.py'),
         'description':'Add soft SE3 root and joint residuals on three overlap frames; weight is experimental, not a paper parameter; all old objectives remain.'}
    save(HERE/'manifest.json',json.dumps(doc,indent=2));print(json.dumps(doc),flush=True)

if __name__=='__main__':main()
