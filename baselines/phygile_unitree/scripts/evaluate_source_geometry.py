"""Explicit diagnostic wrapper around the same first-episode evaluator."""
from pathlib import Path
import argparse,json,runpy,sys
p=argparse.ArgumentParser(add_help=False);p.add_argument('--source-urdf',type=Path,required=True);p.add_argument('--base-scripts',type=Path,required=True)
a,remaining=p.parse_known_args();sys.path.insert(0,str(a.base_scripts))
import mjlab_task
from source_geometry_profile import source_collision_factory
import source_provenance
capture=source_provenance.capture
def bound_capture(output,roots,inputs=()):
    return capture(output,[*roots,Path(__file__).parent],[*inputs,__file__,Path(__file__).with_name('source_geometry_profile.py'),a.source_urdf])
source_provenance.capture=bound_capture
original=mjlab_task.make_config;provenance={}
def make(*args,**kwargs):
    cfg=original(*args,**kwargs);robot=cfg.scene.entities['robot']
    factory,record=source_collision_factory(robot.spec_fn,a.source_urdf);robot.spec_fn=factory
    robot.collisions=();provenance.update(record)
    return cfg
mjlab_task.make_config=make
sys.argv=[str(a.base_scripts/'mjlab_evaluate.py')]+remaining
runpy.run_path(sys.argv[0],run_name='__main__')
out=Path(remaining[remaining.index('--output')+1]);receipt=json.loads((out/'receipt.json').read_text(encoding='utf-8'))
receipt.update(geometry_profile='source_urdf_capsule_primitives',geometry_provenance=provenance,
 note='Same first-episode frozen-policy protocol with explicit source collision geometry. Default official geometry is preserved separately, and no performance improvement is presumed.')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
