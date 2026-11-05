"""Raise only the soft limit penalty in a new isolated helper; no clipping."""
import hashlib,json
from pathlib import Path
R=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');src=R/'gmr29_overlap_anchor_helper_20261103/retarget_helpers/_utils.py';out=R/'gmr29_anchor_limit_v2_helper_20261103'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=src.read_text();old='''        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            100.0,
        ),''';new=old.replace('            100.0,','            1000.0,')
if s.count(old)!=1:raise ValueError('unexpected limit penalty source')
s=s.replace(old,new);out.mkdir(exist_ok=False);d=out/'retarget_helpers';d.mkdir();(d/'__init__.py').write_text('');p=d/'_utils.py';p.write_text(s)
with (out/'provenance.json').open('x') as f:json.dump({'parent_helper_sha256':sha(src),'helper_sha256':sha(p),'changes':'increase soft declared-limit penalty from 100 to 1000; no clipping','training_ready':False},f,indent=2)
print(sha(p))
