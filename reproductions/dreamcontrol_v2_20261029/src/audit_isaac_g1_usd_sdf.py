"""Independent raw-layer Sdf inventory, without composing a stage or running Sim."""
import hashlib,json
from pathlib import Path
from pxr import Sdf
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');PRIMARY=ROOT/'isaac_g1_usd_static_20261030';OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    r=json.loads((PRIMARY/'report.json').read_text());usd=Path(r['usd_path']);assert sha(usd)==r['usd_sha256'] and sha(PRIMARY/'audit.py')==r['script_sha256']
    layer=Sdf.Layer.FindOrOpen(str(usd));assert layer
    paths=[];layer.Traverse('/',lambda path:paths.append(path))
    collision_schemas=[];collision_properties=[];containers=[];not_empty=[];external=[];variant_paths=[];jointtypes={}
    for path in paths:
        spec=layer.GetObjectAtPath(path)
        if isinstance(spec,Sdf.PrimSpec):
            if spec.HasInfo('apiSchemas'):
                names=spec.GetInfo('apiSchemas').GetAddedOrExplicitItems()
                if any('CollisionAPI' in x for x in names):collision_schemas.append(str(path))
            if spec.name=='collisions':
                containers.append(str(path))
                if len(spec.nameChildren)>0:not_empty.append(str(path))
            for key in ('references','payload'):
                if spec.HasInfo(key):
                    value=spec.GetInfo(key)
                    if value.GetAddedOrExplicitItems():external.append({'path':str(path),'field':key,'items':[str(x) for x in value.GetAddedOrExplicitItems()]})
            if spec.typeName in ('PhysicsRevoluteJoint','PhysicsFixedJoint'):jointtypes[str(path)]=spec.typeName
        elif path.IsPropertyPath() and 'collision' in path.name.lower():collision_properties.append(str(path))
        if path.ContainsPrimVariantSelection():variant_paths.append(str(path))
    assert len(containers)==r['collision_container_count'] and len(not_empty)==len(containers)-r['empty_collision_container_count']
    assert r['collision_api_count']==0
    assert sum(v=='PhysicsRevoluteJoint' for v in jointtypes.values())==41 and sum(v=='PhysicsFixedJoint' for v in jointtypes.values())==2
    result={'usd_sha256':sha(usd),'primary_report_sha256':sha(PRIMARY/'report.json'),'audit_script_sha256':sha(Path(__file__)),'spec_paths_checked':len(paths),
       'raw_collision_api_specs':collision_schemas,'collision_containers':len(containers),'nonempty_collision_containers':not_empty,'external_reference_or_payload_specs':external,'variant_selection_spec_paths':len(variant_paths),
       'authored_collision_named_properties':collision_properties,'revolute_joint_specs':41,'fixed_joint_specs':2,
       'limitations':'Checks only this one USD root layer at rest, not Isaac Lab runtime scene, plugin-generated or runtime-authored shapes, physical response or collision filters. Other launch cwd could resolve another file.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))
if __name__=='__main__':main()
