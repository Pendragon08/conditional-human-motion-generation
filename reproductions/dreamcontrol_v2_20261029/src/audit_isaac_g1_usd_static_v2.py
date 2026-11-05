"""Read-only USD/physics schema inventory; does not start Isaac Sim or PhysX."""
import ast,hashlib,json,re
from pathlib import Path
from collections import Counter
from pxr import Usd,UsdPhysics,Sdf
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');OUT=Path(__file__).resolve().parent
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
USD=TRAIN/'HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof_with_hands_min_collisions_flat_white.usd'
UNITREE=TRAIN/'source/isaaclab_assets/isaaclab_assets/robots/unitree.py'
MOTION=TRAIN/'source/isaaclab_tasks/isaaclab_tasks/utils/motion_lib/motion_lib_base.py'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    tree=ast.parse(MOTION.read_text());nodes=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='JointNamesOrder' for t in n.targets)];assert len(nodes)==1;order=ast.literal_eval(nodes[0])
    before=sha(USD);stage=Usd.Stage.Open(str(USD));assert stage and stage.GetDefaultPrim().IsValid()
    prims=list(Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()));joints=[p for p in prims if p.IsA(UsdPhysics.Joint)];revolute=[p.GetName() for p in joints if p.IsA(UsdPhysics.RevoluteJoint)];fixed=[p.GetName() for p in joints if p.IsA(UsdPhysics.FixedJoint)]
    if len(revolute)!=len(set(revolute)):raise ValueError('duplicate revolute joint names')
    missing=[n for n in order if n not in revolute];left=[n for n in revolute if re.fullmatch(r'left_hand.*',n)];right=[n for n in revolute if re.fullmatch(r'right_hand.*',n)]
    collision_api=[str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)]
    applied_collision=[str(p.GetPath()) for p in prims if any('CollisionAPI' in a for a in p.GetAppliedSchemas())]
    collision_containers=[p for p in prims if p.GetName()=='collisions'];empty=[str(p.GetPath()) for p in collision_containers if len(p.GetFilteredChildren(Usd.TraverseInstanceProxies()))==0]
    authored_collision_attrs=[]
    for p in prims:
        for a in p.GetAuthoredAttributes():
            if 'collision' in a.GetName().lower():authored_collision_attrs.append({'prim':str(p.GetPath()),'attribute':a.GetName(),'value':str(a.Get())})
    variantsets=[{'prim':str(p.GetPath()),'sets':p.GetVariantSets().GetNames()} for p in prims if p.GetVariantSets().GetNames()]
    layers=[l.realPath for l in stage.GetUsedLayers() if l.realPath]
    if sha(USD)!=before or len(order)!=27 or missing or len(left)!=7 or len(right)!=7:raise ValueError('asset or joint coverage mismatch')
    indices=[revolute.index(name) for name in order]
    output={'usd_path':str(USD),'usd_sha256':before,'unitree_py_sha256':sha(UNITREE),'motion_py_sha256':sha(MOTION),'script_sha256':sha(Path(__file__)),
        'usd_version':list(Usd.GetVersion()),'default_prim':str(stage.GetDefaultPrim().GetPath()),'used_real_layers':layers,'variant_sets':variantsets,
        'stage_prim_types':dict(Counter(p.GetTypeName() for p in prims)), 'revolute_joint_names_in_usd_traversal':revolute,'fixed_joint_names':fixed,
        'body_joint_names_found':len(order)-len(missing),'body_joint_indices_in_usd_traversal':indices,'left_hand_joint_names':left,'right_hand_joint_names':right,
        'collision_api_count':len(collision_api),'collision_api_paths':collision_api,'collision_enabled_true_count':sum(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is True for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)),'collision_instance_proxy_count':sum(p.IsInstanceProxy() for p in prims if p.HasAPI(UsdPhysics.CollisionAPI)),'applied_any_collision_schema_paths':applied_collision,
        'collision_container_count':len(collision_containers),'empty_collision_container_count':len(empty),'empty_collision_container_paths':empty,
        'authored_collision_named_attributes':authored_collision_attrs,
        'limitations':'Inspects the local USD resolved under the Training working directory; runtime cwd may resolve a different relative asset. Includes instance proxies. Read-only open USD stage, no Isaac Sim spawn, runtime edits, PhysX contact behavior, collision filtering or dynamics. Traversal order is not PhysX articulation order.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps({'revolute':len(revolute),'fixed':fixed,'body':len(order)-len(missing),'left':len(left),'right':len(right),'collision_api':len(collision_api),'containers':len(collision_containers),'empty':len(empty),'variants':len(variantsets),'layers':layers}))
if __name__=='__main__':main()
