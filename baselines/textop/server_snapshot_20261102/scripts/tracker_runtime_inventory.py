"""Read package/runtime facts only; no simulation application or optimization."""
import hashlib,importlib.metadata,importlib.util,json,pathlib,sys
ROOT=pathlib.Path('/data/code/ouyanghao/oyh/textop-reproduction-20261024')
spec=json.loads((ROOT/'configs/tracker-env-spec.json').read_text())
versions={}
for key in ['torch','rsl-rl-lib','numpy','gymnasium','isaacsim','isaaclab']:
    try:versions[key]=importlib.metadata.version(key)
    except importlib.metadata.PackageNotFoundError:versions[key]=None
modules={}
for name in ['torch','rsl_rl','isaaclab','isaaclab_rl','textop_tracker']:
    module=importlib.util.find_spec(name);modules[name]=module.origin if module else None
out={'python':sys.version,'python_executable':sys.executable,'versions':versions,'modules':modules,
     'spec_canonical_hash':hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8],
     'simulation_started':False,'training_started':False}
(ROOT/'receipts/tracker_runtime_inventory.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
