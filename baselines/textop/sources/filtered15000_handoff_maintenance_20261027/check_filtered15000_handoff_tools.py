"""New small CPU primitive/static checks; no replay audit or corpus execution."""
import ast,array,hashlib,json,math,pathlib,sys
W=pathlib.Path(__file__).resolve().parents[2];B=W/'work/phygile_reproduction_20261017/conditional-human-motion-generation/baselines/textop'
O=B/'results/filtered15000_handoff_tools_20261027';O.mkdir(exist_ok=True)
proof=O/'cpu_static_v1.json';assert not proof.exists()
files=['scripts/audit_teacher15000_full_result.py','scripts/supervise_teacher15000_full_audit.py','scripts/filtered15000_inputs.py',
 'scripts/build_filtered15000_corpus.py','scripts/supervise_filtered15000_corpus.py','scripts/run_filtered15000_vae.py','scripts/supervise_filtered15000_vae.py',
 'provider_tools/launch_filtered15000_handoff.py']
trees={n:ast.parse((B/n).read_text(encoding='utf-8')) for n in files}
# Exercise only the new byte-level validation function. Synthetic bytes never
# become a source, trial result, accepted corpus or training state.
node=next(x for x in trees[files[0]].body if isinstance(x,ast.FunctionDef) and x.name=='validate_pose_bytes')
namespace=dict(sys=sys,array=array,math=math);exec(compile(ast.Module(body=[node],type_ignores=[]),'new_pose_byte_validator','exec'),namespace)
validate=namespace['validate_pose_bytes'];row=[0.]*36;row[3]=1.;valid=array.array('f',row*3).tobytes();validate(valid,3)
checks=[]
for label,payload,n in [('wrong_count',valid,2),('short_buffer',valid[:-1],3),
 ('nonfinite',array.array('f',[float('nan')]+row[1:]).tobytes(),1),('bad_quaternion',array.array('f',[0.]*36).tobytes(),1)]:
 try:validate(payload,n)
 except AssertionError:checks.append(label)
 else:raise AssertionError('Invalid synthetic bytes accepted: '+label)
sys.path.insert(0,str(B/'scripts'));from filtered15000_inputs import select_prepared_rows
rows=[dict(canonical_source='synthetic_'+str(i),split='train' if i%2 else 'val',frames50=40+i,dof=29,features=69) for i in range(8278)]
selection={rows[i]['canonical_source']:dict(global_index=i,frames=rows[i]['frames50'],split=rows[i]['split']) for i in [7,2]}
selected=select_prepared_rows(rows,selection);assert [x['canonical_source'] for x in selected]==['synthetic_2','synthetic_7']
bad=dict(selection);bad['not_in_library']=dict(global_index=8278,frames=40,split='val')
try:select_prepared_rows(rows,bad)
except AssertionError:checks.append('unknown_source')
else:raise AssertionError('Unknown source accepted')
for file in [files[0],'scripts/filtered15000_inputs.py','scripts/supervise_teacher15000_full_audit.py']:
 for n in ast.walk(trees[file]):
  if isinstance(n,ast.Import):mods=[x.name.split('.')[0] for x in n.names]
  elif isinstance(n,ast.ImportFrom):mods=[(n.module or '').split('.')[0]]
  else:continue
  assert not set(mods)&{'torch','numpy','isaaclab','gymnasium','textop_tracker'},(file,mods)
for p,key in [('configs/teacher15000-full-result-audit-spec.json','launch_enabled'),('configs/filtered15000-corpus-spec.json','build_authorized'),('configs/filtered15000-vae-spec.json','witness_authorized')]:assert json.loads((B/p).read_text(encoding='utf-8'))[key] is False
assert json.loads((B/'configs/filtered15000-vae-spec.json').read_text(encoding='utf-8'))['train_authorized'] is False
result=dict(status='NEW_HANDOFF_CPU_PRIMITIVE_AND_STATIC_CHECKS_PASSED',synthetic_pose_samples=3,invalid_inputs_rejected=checks,
 selected_source_order_checked=True,cpu_audit_has_no_direct_torch_sdk_import=True,new_specs_all_disabled=True,
 actual_closed_replay_audits=0,actual_filtered_corpus_runs=0,actual_generator_runs=0,new_gpu_contexts=0,
 source_sha256={n:hashlib.sha256((B/n).read_bytes()).hexdigest() for n in files})
proof.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
