"""Draft process-local loader adaptation to retain one additional real tail pose.

Nothing is patched unless the caller explicitly installs the transformed
method in an isolated future job after authenticating the source and lock.
"""
import ast,copy,hashlib

def transformed_method(source,expected_sha256):
 if not isinstance(source,bytes) or hashlib.sha256(source).hexdigest()!=expected_sha256:raise ValueError('loader source hash')
 tree=ast.parse(source);classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SnapMogenLoader']
 if len(classes)!=1:raise ValueError('loader class')
 methods=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='_load_sample']
 if len(methods)!=1:raise ValueError('loader method')
 method=copy.deepcopy(methods[0]);changed=[]
 for n in ast.walk(method):
  if not isinstance(n,ast.Assign) or len(n.targets)!=1 or not isinstance(n.targets[0],ast.Attribute):continue
  t=n.targets[0];v=n.value
  if not (isinstance(t.value,ast.Name) and t.value.id=='bvh_data' and t.attr in ('pos','quats')):continue
  if not (isinstance(v,ast.Subscript) and isinstance(v.value,ast.Attribute) and isinstance(v.value.value,ast.Name) and v.value.value.id=='bvh_data' and v.value.attr==t.attr and isinstance(v.slice,ast.Slice)):raise ValueError('unexpected BVH slice')
  s=v.slice
  if not (isinstance(s.lower,ast.Constant) and s.lower.value==1 and isinstance(s.upper,ast.UnaryOp) and isinstance(s.upper.op,ast.USub) and isinstance(s.upper.operand,ast.Constant) and s.upper.operand.value==3 and s.step is None):raise ValueError('unexpected BVH boundaries')
  s.upper.operand.value=2;changed.append(t.attr)
 if sorted(changed)!=['pos','quats']:raise ValueError('expected exactly two source slices')
 module=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(module)
 return compile(module,'<snapmogen-keep-one-real-tail-frame>','exec')

def install(loader_module,source,expected_sha256):
 original=loader_module.SnapMogenLoader._load_sample;namespace=dict(loader_module.__dict__)
 exec(transformed_method(source,expected_sha256),namespace)
 replacement=namespace['_load_sample'];loader_module.SnapMogenLoader._load_sample=replacement
 return original
