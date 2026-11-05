"""Check reward witness provenance and statically inspect normalization guards."""
import ast
import hashlib
import json
import sys
from pathlib import Path
TRAIN=Path('/data/code/ouyanghao/oyh/conditional-human-motion-generation/third_party/DreamControl/Training')
PICK=TRAIN/'source/isaaclab_tasks/isaaclab_tasks/manager_based/interactive_motion_tracking/g1/motion_tracking_pick_env.py'
MATH=TRAIN/'source/isaaclab/isaaclab/utils/math.py';ROT=TRAIN/'isaac_utils/isaac_utils/rotations.py';MATHS=TRAIN/'isaac_utils/isaac_utils/maths.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=Path(sys.argv[1]);r=json.loads(report.read_text())
for name,p in [('pick',PICK),('isaac_math',MATH),('rotations',ROT),('rotations_math',MATHS)]:assert sha(p)==r[name+'_sha256']
tree=ast.parse(PICK.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='target_orientation_error')
div=[]
for x in ast.walk(function):
    if isinstance(x,ast.BinOp) and isinstance(x.op,ast.Div):div.append(ast.unparse(x))
assert 'a_axis / torch.norm(a_axis, dim=1, keepdim=True)' in div
assert 'b_axis / torch.norm(b_axis, dim=1, keepdim=True)' in div
tracking=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='TRACKING')
assert ast.literal_eval(tracking.value) is True
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='G1Rewards')
rewards=[n for n in ast.walk(cls) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='target_orientation_error' for t in n.targets)]
assert len(rewards)==1
assert any(k.arg=='weight' and ast.literal_eval(k.value)==-1.0 for k in rewards[0].value.keywords)
assert r['results']['finite_false']==[True,False,False,True]
assert r['results']['finite_true']==[True,False,False,True]
out={'status':'passed','witness_sha256':sha(report),'pick_sha256':sha(PICK),'reward_weight':-1.0,'tracking_source_flag':True,'normalization_expressions':div,'script_sha256':sha(Path(__file__)),'limitations':'Static AST and source hashes plus witness result consistency, not independent runtime Isaac simulation or proof real target configurations occur.'}
with Path(sys.argv[2]).open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
