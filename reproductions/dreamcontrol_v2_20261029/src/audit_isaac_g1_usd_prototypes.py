"""Independently map raw Sdf prototype collision specs to authored instance paths."""
import hashlib,json
from pathlib import Path
from pxr import Sdf
ROOT=Path('/data/code/ouyanghao/oyh/dreamcontrol-v2-reproduction-20261029');P=ROOT/'isaac_g1_usd_static_v2_20261030';OUT=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    doc=json.loads((P/'report.json').read_text());usd=Path(doc['usd_path']);assert sha(usd)==doc['usd_sha256'] and sha(P/'audit.py')==doc['script_sha256']
    layer=Sdf.Layer.FindOrOpen(str(usd));paths=[];layer.Traverse('/',lambda path:paths.append(path));raw=[];instances=[]
    for path in paths:
        spec=layer.GetObjectAtPath(path)
        if not isinstance(spec,Sdf.PrimSpec):continue
        if spec.HasInfo('apiSchemas') and 'PhysicsCollisionAPI' in spec.GetInfo('apiSchemas').GetAddedOrExplicitItems():raw.append(str(path))
        if spec.name=='collisions':
            refs=spec.GetInfo('references').GetAddedOrExplicitItems();assert len(refs)==1 and refs[0].assetPath==''
            instances.append((str(path),str(refs[0].primPath)))
    expanded=[]
    for parent,proto in instances:
        matching=[parent+p[len(proto):] for p in raw if p.startswith(proto+'/')]
        assert matching,(parent,proto)
        expanded.extend(matching)
    assert len(instances)==22 and len(raw)==32 and len(expanded)==len(set(expanded))==32
    assert sorted(expanded)==sorted(doc['collision_api_paths']) and doc['collision_instance_proxy_count']==32
    out={'usd_sha256':sha(usd),'stage_report_sha256':sha(P/'report.json'),'audit_script_sha256':sha(Path(__file__)),
         'sdf_specs_checked':len(paths),'internal_reference_count':len(instances),'prototype_collision_api_specs':len(raw),'expanded_collision_paths':len(expanded),
         'independent_raw_prototype_mapping_matches_stage':True,
         'limitations':'Proves collision APIs are authored via internal USD instances, not that runtime PhysX uses or filters them as intended. No simulator, physical contact, policy or PPO.'}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out))
if __name__=='__main__':main()
